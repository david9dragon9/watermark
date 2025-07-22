from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class Arguments:
    root_dir: str
    seed: int

    # Dataset
    dataset_type: str
    file_type: str
    data_files: str
    dataset_config_path: str

    # Model
    model_name: str
    rl_model_name: Optional[str] = None
    adapter_path: Optional[str] = None
    model_type: str = "sequence_classification"
    num_labels: Optional[int] = None
    max_length: Optional[int] = 8192
    eval_max_length: Optional[int] = None

    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0

    target_modules: Optional[List[str]] = None
    target_modules_custom_phrase: Optional[str] = None

    add_pad_token: bool = True
    add_cls_token: bool = False

    embedding_selection: str = (
        "last_token"  # choices: ["last_token", "pooled", "cls_token"]
    )

    target_data_files: Optional[str] = None
    shuffle_target_dataset: bool = False

    eval_dataset_type: Optional[str] = None
    eval_file_type: Optional[str] = None
    eval_data_files: Optional[str] = None
    eval_dataset_config_path: Optional[str] = None

    # Training
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    epoch_save_freq: int = 1
    step_save_freq: int = 1000
    precision: str = "float32"  # choices: ["float32", "float16", "bfloat16"]
    clip_grad_norm: Optional[float] = None

    train_mode: str = (
        "sequence_prediction"  # choices: ["sequence_prediction", "domain_adaptation"]
    )
    eval_mode: str = (
        "sequence_prediction"  # choices: ["sequence_prediction", "domain_adaptation"]
    )
    da_config_path: Optional[str] = None

    use_wandb: bool = False
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None

    mode: str = "train"  # choices: ["train", "eval", "both"]
    target_batch_size: Optional[int] = None
    eval_batch_size: int = 32
    epoch_eval_freq: int = 1
    step_eval_freq: int = 1000

    metric_names: List[str] = field(default_factory=list)
    eval_metadata_path: Optional[str] = None

    # Loss
    loss_type: str = (
        "cross_entropy"  # choices: ["cross_entropy", "mse", "binary_cross_entropy"]
    )
    target_loss_weight: float = 0.0
