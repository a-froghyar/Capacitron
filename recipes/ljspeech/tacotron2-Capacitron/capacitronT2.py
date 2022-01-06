import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CapacitronVAEConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

output_path = "/home/big-boy/Models/CPR/"

# Using LJSpeech like dataset processing for the blizzard dataset
dataset_config = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    path="/home/big-boy/Data/blizzard2013/segmented/",
)

audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=True,
    mel_fmin=80.0,
    mel_fmax=12000,
    spec_gain=20.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
    min_level_db=-100,
)

# Using the standard Capacitron config
capacitron_config = CapacitronVAEConfig(capacitron_VAE_loss_alpha=1.0)

config = Tacotron2Config(
    run_name="Capacitron-T2-DCA-only",
    audio=audio_config,
    capacitron_vae=capacitron_config,
    use_capacitron_vae=True,
    batch_size=95,
    eval_batch_size=16,
    num_loader_workers=12,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=15,
    ga_alpha=0.0,
    r=2,
    attention_type="dynamic_convolution",
    double_decoder_consistency=False,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    use_espeak_phonemes=True,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    stopnet_pos_weight=15,
    print_step=500,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    min_seq_len=1,
    max_seq_len=100,
    lr=1e-3,
    lr_scheduler="StepwiseGradualLR",
    lr_scheduler_params={
        "gradual_learning_rates": [
            [0, 1e-3],
            [2e4, 5e-4],
            [4e5, 3e-4],
            [6e4, 1e-4],
            [8e4, 5e-5],
        ]
    },
    # Need to experiment with these below for capacitron
    dashboard_logger="wandb",
    loss_masking=False,
    decoder_loss_alpha=1.0,
    postnet_loss_alpha=1.0,
    postnet_diff_spec_alpha=0.0,
    decoder_diff_spec_alpha=0.0,
    decoder_ssim_alpha=0.0,
    postnet_ssim_alpha=0.0,
)

ap = AudioProcessor(**config.audio.to_dict())

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

model = Tacotron2(config, speaker_manager=None)

trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)

# 🚀
trainer.fit()