import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

class CapacitronVAE(nn.Module):
    """Effective Use of Variational Embedding Capacity for prosody transfer.
    See https://arxiv.org/abs/1906.03402 """

    def __init__(self,
                 num_mel,
                 capacitron_embedding_dim,
                 encoder_output_dim=256,
                 reference_encoder_out_dim=128,
                 speaker_embedding_dim=None,
                 text_summary_embedding_dim=None,
                 use_iaf=True,
                 iaf_depth=2,
                 iaf_n_blocks=5,
                 iaf_h_dim=128):
        super().__init__()
        # Init distributions
        self.prior_distribution = MVN(torch.zeros(capacitron_embedding_dim), torch.eye(capacitron_embedding_dim))
        self.approximate_posterior_distribution = None
        # define output ReferenceEncoder dim to the capacitron_embedding_dim
        self.encoder = ReferenceEncoder(num_mel, out_dim=reference_encoder_out_dim)

        # Init beta, the lagrange-like term for the KL distribution
        self.beta = torch.nn.Parameter(torch.log(torch.exp(torch.Tensor([1.0])) - 1), requires_grad=True)
        mlp_input_dimension = reference_encoder_out_dim

        if text_summary_embedding_dim is not None:
            self.text_summary_net = TextSummary(text_summary_embedding_dim, encoder_output_dim=encoder_output_dim)
            mlp_input_dimension += text_summary_embedding_dim
        if speaker_embedding_dim is not None:
            # TODO: Figure out what to do with speaker_embedding_dim
            mlp_input_dimension += speaker_embedding_dim

        self.post_encoder_mlp = PostEncoderMLP(mlp_input_dimension, capacitron_embedding_dim)

        # Use inverse autoregressive flow in place for the approximate posterior
        self.use_iaf = use_iaf
        self.iaf_depth = iaf_depth
        self.iaf_n_blocks = iaf_n_blocks

        if self.use_iaf:
            self.epsilon_distribution = MVN(torch.zeros(capacitron_embedding_dim), torch.eye(capacitron_embedding_dim))
            self.post_encoder_iaf_mlp = PostEncoderIAFMLP(mlp_input_dimension, capacitron_embedding_dim)
            # TODO put IAF depth into config
            flows = [IAF(dim=capacitron_embedding_dim, parity=i % 2) for i in range(6)]
            self.flow_model = NormalizingFlowModel(self.epsilon_distribution, flows)

    def forward(self, reference_mel_info=None, text_info=None, speaker_embedding=None):
        iaf_kl_term = None
        # Use reference
        if reference_mel_info is not None:
            reference_mels = reference_mel_info[0] # [batch_size, num_frames, num_mels]
            mel_lengths = reference_mel_info[1] # [batch_size]
            enc_out = self.encoder(reference_mels, mel_lengths)

            # concat speaker_embedding and/or text summary embedding
            if text_info is not None:
                text_inputs = text_info[0] # [batch_size, num_characters, num_embedding]
                input_lengths = text_info[1]
                text_summary_out = self.text_summary_net(text_inputs, input_lengths).to(reference_mels.device)
                enc_out = torch.cat([enc_out, text_summary_out], dim=-1)
            if speaker_embedding is not None:
                enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)

            # Feed the output of the ref encoder and information about text/speaker into
            # an MLP to produce the parameteres for the approximate poterior distributions
            mu, sigma = self.post_encoder_mlp(enc_out)
            # convert to cpu because prior_distribution was created on cpu
            mu = mu.cpu()
            sigma = sigma.cpu()
            # Sample from the posterior: z ~ q(z|x)
            if self.use_iaf:
                if reference_mels.shape[0] != 1:
                    enc_out = self.post_encoder_iaf_mlp(enc_out).to(mu.device)
                    zs, prior_logprob, log_det = self.flow_model.forward(enc_out, mu, sigma)
                    iaf_kl_term = prior_logprob + log_det
                    iaf_kl_term = - iaf_kl_term
                    VAE_embedding = zs[-1]
                else:
                    enc_out = self.post_encoder_iaf_mlp(enc_out)
                    zs, *_ = self.flow_model.sample(1, enc_out, mu, sigma)
                    VAE_embedding = zs[-1].unsqueeze(0)

            else:
                self.approximate_posterior_distribution = MVN(mu, torch.diag_embed(sigma))
                VAE_embedding = self.approximate_posterior_distribution.rsample()

        # Infer from the model, bypasses encoding
        else:
            # Sample from the prior: z ~ p(z)
            VAE_embedding = self.prior_distribution.sample().unsqueeze(0)

        # reshape to [batch_size, 1, capacitron_embedding_dim]
        return VAE_embedding.unsqueeze(1), self.approximate_posterior_distribution, self.prior_distribution, self.beta, iaf_kl_term

class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.
    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, out_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(2, 2)) for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.training = False
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features=filter_size)
            for filter_size in filters[1:]
        ])

        post_conv_height = self.calculate_post_conv_height(
            num_mel, 3, 2, 2, num_layers)
        self.recurrence = nn.LSTM(
            input_size=filters[-1] * post_conv_height,
            hidden_size=out_dim,
            batch_first=True,
            bidirectional=False)

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel) # [batch_size, num_channels==1, num_frames, num_mel]
        valid_lengths = input_lengths.float() # [batch_size]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            # Create the post conv width mask based on the valid lengths of the output of the convolution.
            # The valid lengths for the output of a convolution on varying length inputs is
            # ceil(input_length/stride) + 1 for stride=3 and padding=2
            # For example (kernel_size=3, stride=2, padding=2):
            # 0 0 x x x x x 0 0 -> Input = 5, 0 is zero padding, x is valid values coming from padding=2 in conv2d
            # _____
            #   x _____
            #       x _____
            #           x  ____
            #               x
            # x x x x -> Output valid length = 4
            # Since every example in te batch is zero padded and therefore have separate valid_lengths,
            # we need to mask off all the values AFTER the valid length for each example in the batch.
            # Otherwise, the convolutions create noise and a lot of not real information
            valid_lengths = (valid_lengths/2).float()
            valid_lengths = torch.ceil(valid_lengths).to(dtype=torch.int64) + 1 # 2 is stride -- size: [batch_size]
            post_conv_max_width = x.size(2)

            mask = torch.arange(post_conv_max_width).to(inputs.device).expand(len(valid_lengths), post_conv_max_width) < valid_lengths.unsqueeze(1)
            mask = mask.expand(1, 1, -1, -1).transpose(2, 0).transpose(-1, 2) # [batch_size, 1, post_conv_max_width, 1]
            x = x*mask

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]

        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        post_conv_input_lengths = valid_lengths
        packed_seqs = nn.utils.rnn.pack_padded_sequence(x, post_conv_input_lengths.tolist(), batch_first=True, enforce_sorted=False) # dynamic rnn sequence padding
        self.recurrence.flatten_parameters()
        _, (ht, _) = self.recurrence(packed_seqs)
        last_output = ht[-1]

        return last_output.to(inputs.device) # [B, 128]

    @ staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad,
                                   n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height

class TextSummary(nn.Module):
    def __init__(self, embedding_dim, encoder_output_dim):
        super().__init__()
        self.lstm = nn.LSTM(encoder_output_dim, # text embedding dimension from the text encoder
                            embedding_dim, # fixed length output summary the lstm creates from the input
                            batch_first=True,
                            bidirectional=False)

    def forward(self, inputs, input_lengths):
        # TODO deal with inference - input lengths is not a tensor but just a single number
        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        packed_seqs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.tolist(), batch_first=True, enforce_sorted=False) # dynamic rnn sequence padding
        self.lstm.flatten_parameters()
        _, (ht, _) = self.lstm(packed_seqs)
        last_output = ht[-1]
        return last_output

class PostEncoderMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        modules = [
            nn.Linear(input_size, hidden_size), # Hidden Layer
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2)] # Output layer twice the size for mean and variance
        self.net = nn.Sequential(*modules)
        self.softplus = nn.Softplus()

    def forward(self, _input):
        mlp_output = self.net(_input)
        # The mean parameter is unconstrained
        mu = mlp_output[:, :self.hidden_size]
        # The standard deviation must be positive. Parameterise with a softplus
        sigma = self.softplus(mlp_output[:, self.hidden_size:])
        return mu, sigma

class PostEncoderIAFMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Output layer single projection for IAF
        self.linear_layer = nn.Linear(input_size, hidden_size) # Hidden Layer
        # self.elu_layer = nn.ELU()

    def forward(self, _input):
        mlp_output = self.linear_layer(_input)
        return mlp_output

class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nh, embedding_dim):
        super().__init__()
        self.net = MADE(nin, nh, num_cond_inputs=embedding_dim, s_act='elu', t_act='elu', pre_exp_tanh=False)

    def forward(self, x, context):
        return self.net(x, context)

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """

    def __init__(self, dim, parity, net_class=ARMLP):
        super().__init__()
        self.dim = dim
        # TODO: 128 is capacitron embedding dim, don't hardcode
        self.net = net_class(dim, dim, 128)
        self.parity = parity

    def forward(self, context, z):
        _device = 'cuda' if torch.cuda.is_available() else context.device

        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        context = context.flip(dims=(1,)) if self.parity else context

        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        s, m = self.net(z.clone().to(_device), context) # clone to avoid in-place op errors if using IAF
        s = torch.sigmoid(s)
        s, m = s.to(z.device), m.to(z.device) # need to parallelise on the GPU

        # z = s * z + (1-s)*m
        z = (z-m)/s
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, context, previous_z):
        # TODO: fix z and x naming
        # we have to decode the x one at a time, sequentially
        _device = 'cuda' if torch.cuda.is_available() else context.device
        log_det = torch.zeros(previous_z.size(0))

        previous_z = previous_z.flip(dims=(1,)) if self.parity else previous_z
        context = context.flip(dims=(1,)) if self.parity else context

        x = previous_z.clone()

        for i in range(self.dim):
            s, m = self.net(x.clone().to(_device), context.clone().to(_device)) # clone to avoid in-place op errors if using IAF
            s = torch.sigmoid(s)
            s, m = s.cpu(), m.cpu() #no need to parallelise on the GPU

            x[:, i] = previous_z[:, i] * s[:, i] + (1 - s[:, i]) * m[:, i]
            log_det += -torch.log(s[:, i])
        return x, log_det

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead,
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 pre_exp_tanh=False):
        super().__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'elu': nn.ELU}

        input_mask = self.get_mask(num_inputs, num_hidden, num_inputs,
                                   mask_type='input')
        hidden_mask = self.get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = self.get_mask(num_hidden, num_inputs, num_inputs,
                                    mask_type='output')

        act_func = activations[s_act]
        self.s_joiner = MaskedLinear(num_inputs, num_hidden, input_mask,
                                     cond_in_features=num_cond_inputs)

        self.s_trunk = nn.Sequential(act_func(),
                                     MaskedLinear(num_hidden, num_hidden,
                                                  hidden_mask), act_func(),
                                     MaskedLinear(num_hidden, num_inputs,
                                                  output_mask))

        act_func = activations[t_act]
        self.t_joiner = MaskedLinear(num_inputs, num_hidden, input_mask,
                                     cond_in_features=num_cond_inputs)

        self.t_trunk = nn.Sequential(act_func(),
                                     MaskedLinear(num_hidden, num_hidden,
                                                  hidden_mask), act_func(),
                                     MaskedLinear(num_hidden, num_inputs,
                                                  output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)

            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)

            # u = (inputs - m) * torch.exp(-a)
            # return u, -a.sum(-1, keepdim=True)
            return a, m

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

    @ staticmethod
    def get_mask(in_features, out_features, in_flow_features, mask_type=None):
        """
        mask_type: input | None | output

        See Figure 1 for a better illustration:
        https://arxiv.org/pdf/1502.03509.pdf
        """
        if mask_type == 'input':
            in_degrees = torch.arange(in_features) % in_flow_features
        else:
            in_degrees = torch.arange(in_features) % (in_flow_features - 1)

        if mask_type == 'output':
            out_degrees = torch.arange(out_features) % in_flow_features - 1
        else:
            out_degrees = torch.arange(out_features) % (in_flow_features - 1)

        return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, initial_z, context, initial_log_det):
        zs = [initial_z]
        log_det = initial_log_det

        for flow in self.flows:
            new_z, ld = flow.forward(context, zs[-1])
            log_det += ld.to(log_det.device)
            zs.append(new_z)
        return zs, log_det

    def backward(self, initial_x, context):
        xs = [initial_x]
        m, _ = initial_x.shape
        log_det = torch.zeros(m)

        for flow in self.flows[::-1]:
            new_x, ld = flow.backward(context, xs[-1])
            log_det += ld
            xs.append(new_x)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, context, initial_mu, initial_sigma):
        epsilon_sample = self.prior.rsample((context.shape[0],))
        initial_z = initial_sigma * epsilon_sample + initial_mu
        pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)

        initial_log_det = - torch.sum(torch.log(initial_sigma) + 0.5 * epsilon_sample**2 + 0.5 * torch.log(2*pi), axis=-1) #initial l

        zs, log_det = self.flow.forward(initial_z, context, initial_log_det)
        prior_logprob = self.prior.log_prob(zs[-1].cpu()).view(context.size(0), -1).sum(1)
        final_density = - log_det

        return zs, prior_logprob, final_density

    # def backward(self, z):
    #     xs, log_det = self.flow.backward(z)
    #     return xs, log_det

    def sample(self, num_samples, context, initial_mu, initial_sigma):
        epsilon_sample = self.prior.sample((num_samples,))
        initial_x = (epsilon_sample - initial_mu) / initial_sigma
        xs, _ = self.flow.backward(initial_x, context)
        return xs
