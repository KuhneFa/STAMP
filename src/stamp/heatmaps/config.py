from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stamp.types import SlideMPP, get_default_device


class HeatmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="Directory to save heatmap outputs")

    feature_dir: Path = Field(description="Directory containing extracted features")
    wsi_dir: Path = Field(description="Directory containing whole slide images")
    checkpoint_path: Path = Field(description="Path to model checkpoint file")

    slide_paths: list[Path] | None = Field(
        default=None,
        description="Specific slide paths to process. If None, processes all slides in wsi_dir",
    )

    device: str = Field(
        default_factory=get_default_device,
        description="Device to use for computation",
    )

    opacity: float = Field(
        default=0.6,
        description="Overlay plot opacity. A value of 0 means transparent and 1 opaque.",
        ge=0,
        le=1,
    )

    topk: int = Field(
        default=0,
        description="Number of top patches to highlight. 0 means no highlighting.",
        ge=0,
    )

    bottomk: int = Field(
        default=0,
        description="Number of bottom patches to highlight. 0 means no highlighting.",
        ge=0,
    )

    default_slide_mpp: SlideMPP | None = Field(
        default=None,
        description="MPP of the slide to use if none can be inferred from the WSI",
    )
