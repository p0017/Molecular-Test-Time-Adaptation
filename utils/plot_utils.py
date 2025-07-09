import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from beartype import beartype
from torch_scatter import scatter_mean
import torch
import seaborn as sns
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@beartype
def solubility_histogram(
    df_train: pd.DataFrame, df_test: pd.DataFrame, save_plots: bool = False
):
    """Plotting a histogram of solubility values for training and test datasets.
    Args:
        df_train (pd.DataFrame): Training dataset containing 'ExperimentalLogS' column.
        df_test (pd.DataFrame): Test dataset containing 'LogS' column.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    assert (
        "ExperimentalLogS" in df_train.columns
    ), "Training Data must contain 'ExperimentalLogS' column"
    assert "LogS" in df_test.columns, "Test Data must contain 'LogS' column"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(
        df_train["ExperimentalLogS"],
        bins=100,
        alpha=0.7,
        label="Training Data (AqSolDBc)",
        color="blue",
        density=False,
    )

    ax.hist(
        df_test["LogS"],
        bins=100,
        alpha=0.7,
        label="Test Data (OChemUnseen)",
        color="orange",
        density=False,
    )

    ax.set_xlabel("Log Solubility")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/solubility_histogram.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/solubility_histogram.pdf", bbox_inches="tight")
    plt.show()


@beartype
def solubility_histogram_shift(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test_1: pd.DataFrame,
    df_test_2: pd.DataFrame,
    save_plots: bool = False,
):
    """Plotting a histogram of solubility values for training and test datasets.
    Args:
        df_train (pd.DataFrame): Training dataset containing 'ExperimentalLogS' column.
        df_val (pd.DataFrame): Validation dataset containing 'ExperimentalLogS' column.
        df_test_1 (pd.DataFrame): Test dataset containing 'LogS' column.
        df_test_2 (pd.DataFrame): Another test dataset containing 'LogS' column.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    assert (
        "ExperimentalLogS" in df_train.columns
    ), "Training Data must contain 'ExperimentalLogS' column"
    assert (
        "ExperimentalLogS" in df_val.columns
    ), "Validation Data must contain 'ExperimentalLogS' column"
    assert "LogS" in df_test_1.columns, "Test Data must contain 'LogS' column"
    assert "LogS" in df_test_2.columns, "Test Data must contain 'LogS' column"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(
        df_train["ExperimentalLogS"],
        bins=100,
        alpha=0.5,
        label="Training Data",
        color="blue",
        density=False,
    )

    ax.hist(
        df_val["ExperimentalLogS"],
        bins=100,
        alpha=0.5,
        label="Validation Data",
        color="orange",
        density=False,
    )

    ax.hist(
        df_test_1["LogS"],
        bins=100,
        alpha=0.5,
        label="20 Atom Test Data",
        color="green",
        density=False,
    )

    ax.hist(
        df_test_2["LogS"],
        bins=100,
        alpha=0.5,
        label="NH2 Test Data",
        color="red",
        density=False,
    )

    ax.set_xlabel("Log Solubility")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            "figures/solubility_histogram_shift.jpg", dpi=300, bbox_inches="tight"
        )
        plt.savefig("figures/solubility_histogram_shift.pdf", bbox_inches="tight")
    plt.show()


@beartype
def atom_count_histogram(
    df_train: pd.DataFrame, df_test: pd.DataFrame, save_plots: bool = False
):
    """Plotting a histogram of the number of atoms in molecules for training and test datasets.
    Args:
        df_train (pd.DataFrame): Training dataset containing 'num_atoms' column.
        df_test (pd.DataFrame): Test dataset containing 'num_atoms' column.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    assert (
        "num_atoms" in df_train.columns
    ), "Training Data must contain 'num_atoms' column"
    assert "num_atoms" in df_test.columns, "Test Data must contain 'num_atoms' column"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(
        df_train["num_atoms"],
        bins=100,
        alpha=0.7,
        label="Training Data (AqSolDBc)",
        color="blue",
        density=True,
    )

    ax.hist(
        df_test["num_atoms"],
        bins=100,
        alpha=0.7,
        label="Test Data (OChemUnseen)",
        color="orange",
        density=True,
    )

    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 120)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/atom_count_histogram.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/atom_count_histogram.pdf", bbox_inches="tight")
    plt.show()


@beartype
def functional_group_histogram(
    df_train: pd.DataFrame, df_test: pd.DataFrame, save_plots: bool = False
):
    """Plotting histograms of functional groups for training and test datasets.
    Args:
        df_train (pd.DataFrame): Training dataset containing functional group columns.
        df_test (pd.DataFrame): Test dataset containing functional group columns.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    functional_groups = [
        "fr_Al_OH",
        "fr_Ar_OH",
        "fr_COO",
        "fr_NH2",
        "fr_amide",
        "fr_ester",
        "fr_ether",
        "fr_halogen",
    ]
    assert all(
        fg in df_train.columns for fg in functional_groups
    ), "Training Data must contain all functional group columns"
    assert all(
        fg in df_test.columns for fg in functional_groups
    ), "Test Data must contain all functional group columns"

    # Create 8 subplots for functional group histograms with shared y-axis per row
    fig, axes = plt.subplots(2, 4, figsize=(8, 6), sharey="row", sharex="col")
    axes = axes.flatten()

    for i, fg in enumerate(functional_groups):
        ax = axes[i]

        # Filter to only include nonzero values
        train_nonzero = df_train[df_train[fg] > 0][fg]
        test_nonzero = df_test[df_test[fg] > 0][fg]

        # Get the range of values for proper binning
        max_val = max(
            train_nonzero.max() if len(train_nonzero) > 0 else 0,
            test_nonzero.max() if len(test_nonzero) > 0 else 0,
        )

        if max_val > 0:
            # Create bins with edges at 0.5, 1.5, 2.5, ... to center each integer value
            bins = np.arange(0.5, max_val + 1.5, 1)

            # Plot histograms for both datasets (only nonzero values)
            ax.hist(
                train_nonzero,
                bins=bins,
                alpha=0.7,
                label="Training Data (AqSolDBc)",
                color="blue",
                density=False,
            )

            ax.hist(
                test_nonzero,
                bins=bins,
                alpha=0.7,
                label="Test Data (OChemUnseen)",
                color="orange",
                density=False,
            )

            # Set x-axis to show only integer values
            ax.set_xticks(range(1, 10))
            ax.set_xlim(0.5, max_val + 0.5)

        ax.set_xlabel(fg.replace("fr_", "").replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 10.5)

        # Only show y-label on leftmost plots
        if i % 4 == 0:
            ax.set_ylabel("Count")

        # Only show legend on first subplot
        if i == 4:
            ax.legend(bbox_to_anchor=(4, -0.4), loc="lower right", ncol=2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            "figures/functional_groups_distribution.jpg", dpi=300, bbox_inches="tight"
        )
        plt.savefig("figures/functional_groups_distribution.pdf", bbox_inches="tight")
    plt.show()


@beartype
def embedding_histogram(
    train_embeddings,
    val_embeddings,
    test_embeddings_20,
    test_embeddings_nh,
    val_embeddings_with_TTA,
    test_embeddings_with_TTA_20,
    test_embeddings_with_TTA_nh,
    save_plots: bool = False,
):
    """Plotting a histogram of embedding values for training and test datasets.

    Args:
        train_embeddings): Training embeddings with shape (n_samples, 16)
        val_embeddings: Validation embeddings with shape (n_samples, 16)
        test_embeddings_20: Test 20-atom embeddings with shape (n_samples, 16)
        test_embeddings_nh: Test NH2 embeddings with shape (n_samples, 16)
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    train_flat = np.array(train_embeddings).flatten()
    val_flat = np.array(val_embeddings).flatten()
    test_20_flat = np.array(test_embeddings_20).flatten()
    test_nh_flat = np.array(test_embeddings_nh).flatten()
    val_with_TTA_flat = np.array(val_embeddings_with_TTA).flatten()
    test_with_TTA_20_flat = np.array(test_embeddings_with_TTA_20).flatten()
    test_with_TTA_nh_flat = np.array(test_embeddings_with_TTA_nh).flatten()

    # Min and max values across all embeddings
    min_val = min(
        train_flat.min(),
        val_flat.min(),
        test_20_flat.min(),
        test_nh_flat.min(),
        val_with_TTA_flat.min(),
        test_with_TTA_20_flat.min(),
        test_with_TTA_nh_flat.min(),
    )
    # Max determined using percentile due to outliers
    percentile = 99.9
    max_val = max(
        np.percentile(train_flat, percentile),
        np.percentile(val_flat, percentile),
        np.percentile(test_20_flat, percentile),
        np.percentile(test_nh_flat, percentile),
        np.percentile(val_with_TTA_flat, percentile),
        np.percentile(test_with_TTA_20_flat, percentile),
        np.percentile(test_with_TTA_nh_flat, percentile),
    )

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    ax[0].hist(
        train_flat,
        bins=100,
        alpha=0.7,
        label="Training Embeddings",
        color="blue",
        density=False,
    )

    ax[1].hist(
        test_20_flat,
        bins=100,
        alpha=0.5,
        label="20 Atom Test Embeddings",
        color="green",
        density=False,
    )

    ax[1].hist(
        test_nh_flat,
        bins=100,
        alpha=0.5,
        label="NH2 Test Embeddings",
        color="red",
        density=False,
    )

    ax[1].hist(
        val_flat,
        bins=100,
        alpha=0.5,
        label="Validation Embeddings",
        color="orange",
        density=False,
    )

    ax[2].hist(
        test_with_TTA_20_flat,
        bins=100,
        alpha=0.5,
        label="20 Atom Test Embeddings with TTA",
        color="green",
        density=False,
    )

    ax[2].hist(
        test_with_TTA_nh_flat,
        bins=100,
        alpha=0.5,
        label="NH2 Test Embeddings with TTA",
        color="red",
        density=False,
    )

    ax[2].hist(
        val_with_TTA_flat,
        bins=100,
        alpha=0.5,
        label="Validation Embeddings with TTA",
        color="orange",
        density=False,
    )

    ax[2].set_xlabel("Embedding Value")

    for axis in ax:
        axis.set_ylabel("Count")
        axis.set_xlim(min_val, max_val)
        axis.set_yscale("log")
        axis.legend()
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/embedding_histogram.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/embedding_histogram.pdf", bbox_inches="tight")
    plt.show()


@beartype
def get_dynamic_limits(
    train_embeddings,
    val_embeddings,
    test_embeddings_1,
    test_embeddings_2,
    lower_percentile: int = 5,
    upper_percentile: int = 95,
):
    """Calculate dynamic limits for plotting embeddings based on percentiles.
    Args:
        train_embeddings (np.ndarray): Embeddings for the training set.
        val_embeddings (np.ndarray): Embeddings for the validation set.
        test_embeddings_1 (np.ndarray): Embeddings for a test set.
        test_embeddings_2 (np.ndarray): Embeddings for another test set.
        lower_percentile (int): Lower percentile for dynamic limits. Defaults to 5.
        upper_percentile (int): Upper percentile for dynamic limits. Defaults to 95.
    Returns:
        tuple: (x_min, x_max, y_min, y_max) for the plot limits.
    """

    assert (
        lower_percentile < upper_percentile
    ), "Lower percentile must be less than upper percentile"
    assert 0 <= lower_percentile <= 100, "Lower percentile must be between 0 and 100"
    assert 0 <= upper_percentile <= 100, "Upper percentile must be between 0 and 100"

    x_min_train, x_max_train = np.percentile(
        train_embeddings[:, 0], [lower_percentile, upper_percentile]
    )
    y_min_train, y_max_train = np.percentile(
        train_embeddings[:, 1], [lower_percentile, upper_percentile]
    )
    x_min_val, x_max_val = np.percentile(
        val_embeddings[:, 0], [lower_percentile, upper_percentile]
    )
    y_min_val, y_max_val = np.percentile(
        val_embeddings[:, 1], [lower_percentile, upper_percentile]
    )
    x_min_test_1, x_max_test_1 = np.percentile(
        test_embeddings_1[:, 0], [lower_percentile, upper_percentile]
    )
    y_min_test_1, y_max_test_1 = np.percentile(
        test_embeddings_1[:, 1], [lower_percentile, upper_percentile]
    )
    x_min_test_2, x_max_test_2 = np.percentile(
        test_embeddings_2[:, 0], [lower_percentile, upper_percentile]
    )
    y_min_test_2, y_max_test_2 = np.percentile(
        test_embeddings_2[:, 1], [lower_percentile, upper_percentile]
    )

    x_max = max(x_max_train, x_max_test_1, x_max_test_2, x_max_val)
    x_min = min(x_min_train, x_min_test_1, x_min_test_2, x_min_val)
    y_max = max(y_max_train, y_max_test_1, y_max_test_2, y_max_val)
    y_min = min(y_min_train, y_min_test_1, y_min_test_2, y_min_val)

    return x_min, x_max, y_min, y_max


@beartype
def solubility_embeddings(
    train_solubility,
    val_solubility,
    test_solubility_1,
    test_solubility_2,
    train_embeddings_tsne: np.ndarray,
    val_embeddings_tsne: np.ndarray,
    test_embeddings_tsne_1: np.ndarray,
    test_embeddings_tsne_2: np.ndarray,
    train_embeddings_umap: np.ndarray,
    val_embeddings_umap: np.ndarray,
    test_embeddings_umap_1: np.ndarray,
    test_embeddings_umap_2: np.ndarray,
    save_plots: bool = False,
):
    """Plotting t-SNE and UMAP projections of embeddings colored by solubility.
    Args:
        train_solubility: Solubility values for the training set.
        val_solubility: Solubility values for the validation set.
        test_solubility_1: Solubility values for a test set.
        test_solubility_2: Solubility values for another test set.
        train_embeddings_tsne (np.ndarray): t-SNE embeddings for the training set.
        val_embeddings_tsne (np.ndarray): t-SNE embeddings for the validation set.
        test_embeddings_tsne_1 (np.ndarray): t-SNE embeddings for a test set.
        test_embeddings_tsne_2 (np.ndarray): t-SNE embeddings for another test set.
        train_embeddings_umap (np.ndarray): UMAP embeddings for the training set.
        val_embeddings_umap (np.ndarray): UMAP embeddings for the validation set.
        test_embeddings_umap_1 (np.ndarray): UMAP embeddings for the test set.
        test_embeddings_umap_2 (np.ndarray): UMAP embeddings for another test set.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Define a consistent colormap for solubility
    cmap = plt.cm.viridis
    norm = plt.Normalize(
        min(
            min(train_solubility),
            min(val_solubility),
            min(test_solubility_1),
            min(test_solubility_2),
        ),
        max(
            max(train_solubility),
            max(val_solubility),
            max(test_solubility_1),
            max(test_solubility_2),
        ),
    )

    # Plot t-SNE projection on the left
    sc1 = ax1.scatter(
        train_embeddings_tsne[:, 0],
        train_embeddings_tsne[:, 1],
        c=train_solubility,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax1.scatter(
        val_embeddings_tsne[:, 0],
        val_embeddings_tsne[:, 1],
        c=val_solubility,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax1.scatter(
        test_embeddings_tsne_1[:, 0],
        test_embeddings_tsne_1[:, 1],
        c=test_solubility_1,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax1.scatter(
        test_embeddings_tsne_2[:, 0],
        test_embeddings_tsne_2[:, 1],
        c=test_solubility_2,
        cmap=cmap,
        norm=norm,
        s=2,
    )

    ax1.set_title("t-SNE Projection")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    # Plot UMAP projection on the right
    sc2 = ax2.scatter(
        train_embeddings_umap[:, 0],
        train_embeddings_umap[:, 1],
        c=train_solubility,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax2.scatter(
        val_embeddings_umap[:, 0],
        val_embeddings_umap[:, 1],
        c=val_solubility,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax2.scatter(
        test_embeddings_umap_1[:, 0],
        test_embeddings_umap_1[:, 1],
        c=test_solubility_1,
        cmap=cmap,
        norm=norm,
        s=2,
    )
    ax2.scatter(
        test_embeddings_umap_2[:, 0],
        test_embeddings_umap_2[:, 1],
        c=test_solubility_2,
        cmap=cmap,
        norm=norm,
        s=2,
    )

    ax2.set_title("UMAP Projection")
    ax2.set_xlabel("UMAP Component 1")
    ax2.set_ylabel("UMAP Component 2")

    # Dynamically adjust the limits for both plots
    x_min_tsne, x_max_tsne, y_min_tsne, y_max_tsne = get_dynamic_limits(
        train_embeddings_tsne,
        val_embeddings_tsne,
        test_embeddings_tsne_1,
        test_embeddings_tsne_2,
    )
    ax1.set_xlim(x_min_tsne, x_max_tsne)
    ax1.set_ylim(y_min_tsne, y_max_tsne)

    x_min_umap, x_max_umap, y_min_umap, y_max_umap = get_dynamic_limits(
        train_embeddings_umap,
        val_embeddings_umap,
        test_embeddings_umap_1,
        test_embeddings_umap_2,
    )
    ax2.set_xlim(x_min_umap, x_max_umap)
    ax2.set_ylim(y_min_umap, y_max_umap)

    fig.tight_layout()
    cbar = fig.colorbar(sc2, ax=[ax2], aspect=50, pad=0.015)
    cbar.set_label("Solubility")

    if save_plots:
        plt.savefig("figures/solubility.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/solubility.pdf", bbox_inches="tight")
    plt.show()


@beartype
def sets_embeddings(
    train_embeddings_tsne: np.ndarray,
    val_embeddings_tsne: np.ndarray,
    test_embeddings_tsne_1: np.ndarray,
    test_embeddings_tsne_2: np.ndarray,
    train_embeddings_umap: np.ndarray,
    val_embeddings_umap: np.ndarray,
    test_embeddings_umap_1: np.ndarray,
    test_embeddings_umap_2: np.ndarray,
    save_plots: bool = False,
):
    """Plotting t-SNE and UMAP projections of embeddings for different sets.
    Args:
        train_embeddings_tsne (np.ndarray): t-SNE embeddings for the training set.
        val_embeddings_tsne (np.ndarray): t-SNE embeddings for the validation set.
        test_embeddings_tsne_1 (np.ndarray): t-SNE embeddings for a test set.
        test_embeddings_tsne_2 (np.ndarray): t-SNE embeddings for another test set.
        train_embeddings_umap (np.ndarray): UMAP embeddings for the training set.
        val_embeddings_umap (np.ndarray): UMAP embeddings for the validation set.
        test_embeddings_umap_1 (np.ndarray): UMAP embeddings for a test set.
        test_embeddings_umap_2 (np.ndarray): UMAP embeddings for another test set.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # t-SNE plot on the left
    ax1.scatter(
        train_embeddings_tsne[:, 0],
        train_embeddings_tsne[:, 1],
        label="Train Set",
        s=2,
        color="blue",
    )
    ax1.scatter(
        val_embeddings_tsne[:, 0],
        val_embeddings_tsne[:, 1],
        label="Validation Set",
        s=2,
        color="orange",
    )
    ax1.scatter(
        test_embeddings_tsne_1[:, 0],
        test_embeddings_tsne_1[:, 1],
        label="20 Atom Test Set",
        s=2,
        color="green",
    )
    ax1.scatter(
        test_embeddings_tsne_2[:, 0],
        test_embeddings_tsne_2[:, 1],
        label="NH2 Test Set",
        s=2,
        color="red",
    )

    # Dynamically adjust the limits for both plots
    x_min_tsne, x_max_tsne, y_min_tsne, y_max_tsne = get_dynamic_limits(
        train_embeddings_tsne,
        val_embeddings_tsne,
        test_embeddings_tsne_1,
        test_embeddings_tsne_2,
    )

    ax1.set_title("t-SNE Projection")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")
    ax1.set_xlim(x_min_tsne, x_max_tsne)
    ax1.set_ylim(y_min_tsne, y_max_tsne)
    fig.legend(bbox_to_anchor=(0.9, -0.05), loc="lower right", ncol=4)

    # UMAP plot on the right
    ax2.scatter(
        train_embeddings_umap[:, 0],
        train_embeddings_umap[:, 1],
        label="Train Set",
        s=2,
        color="blue",
    )
    ax2.scatter(
        val_embeddings_umap[:, 0],
        val_embeddings_umap[:, 1],
        label="Validation Set",
        s=2,
        color="orange",
    )
    ax2.scatter(
        test_embeddings_umap_1[:, 0],
        test_embeddings_umap_1[:, 1],
        label="20 Atom Test Set",
        s=2,
        color="green",
    )
    ax2.scatter(
        test_embeddings_umap_2[:, 0],
        test_embeddings_umap_2[:, 1],
        label="NH2 Test Set",
        s=2,
        color="red",
    )

    # Dynamically adjust the limits for both plots
    x_min_umap, x_max_umap, y_min_umap, y_max_umap = get_dynamic_limits(
        train_embeddings_umap,
        val_embeddings_umap,
        test_embeddings_umap_1,
        test_embeddings_umap_2,
    )

    ax2.set_title("UMAP Projection")
    ax2.set_xlabel("UMAP Component 1")
    ax2.set_ylabel("UMAP Component 2")
    ax2.set_xlim(x_min_umap, x_max_umap)
    ax2.set_ylim(y_min_umap, y_max_umap)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/sets.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/sets.pdf", bbox_inches="tight")
    plt.show()


@beartype
def centroid_embeddings(
    train_embeddings_tsne: np.ndarray,
    val_embeddings_tsne: np.ndarray,
    val_embeddings_with_TTA_tsne: np.ndarray,
    test_embeddings_tsne_1: np.ndarray,
    test_embeddings_tsne_2: np.ndarray,
    test_embeddings_with_TTA_tsne_1: np.ndarray,
    test_embeddings_with_TTA_tsne_2: np.ndarray,
    train_embeddings_umap: np.ndarray,
    val_embeddings_umap: np.ndarray,
    val_embeddings_with_TTA_umap: np.ndarray,
    test_embeddings_umap_1: np.ndarray,
    test_embeddings_umap_2: np.ndarray,
    test_embeddings_with_TTA_umap_1: np.ndarray,
    test_embeddings_with_TTA_umap_2: np.ndarray,
    save_plots: bool = False,
):
    """Plotting t-SNE and UMAP projections of embeddings with centroids for different sets.
    Args:
        train_embeddings_tsne (np.ndarray): t-SNE embeddings for the training set.
        val_embeddings_tsne (np.ndarray): t-SNE embeddings for the validation set.
        val_embeddings_with_TTA_tsne (np.ndarray): t-SNE embeddings for the validation set with TTA.
        test_embeddings_tsne_1 (np.ndarray): t-SNE embeddings for a test set.
        test_embeddings_tsne_2 (np.ndarray): t-SNE embeddings for another test set.
        test_embeddings_with_TTA_tsne_1 (np.ndarray): t-SNE embeddings for a test set with TTA.
        test_embeddings_with_TTA_tsne_2 (np.ndarray): t-SNE embeddings for another test set with TTA.
        train_embeddings_umap (np.ndarray): UMAP embeddings for the training set.
        val_embeddings_umap (np.ndarray): UMAP embeddings for the validation set.
        val_embeddings_with_TTA_umap (np.ndarray): UMAP embeddings for the validation set with TTA.
        test_embeddings_umap_1 (np.ndarray): UMAP embeddings for a test set.
        test_embeddings_umap_2 (np.ndarray): UMAP embeddings for another test set.
        test_embeddings_with_TTA_umap_1 (np.ndarray): UMAP embeddings for a test set with TTA.
        test_embeddings_with_TTA_umap_2 (np.ndarray): UMAP embeddings for another test set with TTA.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    # Calculate centroids
    train_centroid_umap = np.mean(train_embeddings_umap, axis=0)
    val_centroid_umap = np.mean(val_embeddings_umap, axis=0)
    val_tta_centroid_umap = np.mean(val_embeddings_with_TTA_umap, axis=0)
    test_centroid_umap_1 = np.mean(test_embeddings_umap_1, axis=0)
    test_tta_centroid_umap_1 = np.mean(test_embeddings_with_TTA_umap_1, axis=0)
    test_centroid_umap_2 = np.mean(test_embeddings_umap_2, axis=0)
    test_tta_centroid_umap_2 = np.mean(test_embeddings_with_TTA_umap_2, axis=0)

    train_centroid_tsne = np.mean(train_embeddings_tsne, axis=0)
    val_centroid_tsne = np.mean(val_embeddings_tsne, axis=0)
    val_tta_centroid_tsne = np.mean(val_embeddings_with_TTA_tsne, axis=0)
    test_centroid_tsne_1 = np.mean(test_embeddings_tsne_1, axis=0)
    test_tta_centroid_tsne_1 = np.mean(test_embeddings_with_TTA_tsne_1, axis=0)
    test_centroid_tsne_2 = np.mean(test_embeddings_tsne_2, axis=0)
    test_tta_centroid_tsne_2 = np.mean(test_embeddings_with_TTA_tsne_2, axis=0)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # t-SNE plot on the left
    ax1.scatter(
        test_embeddings_tsne_1[:, 0],
        test_embeddings_tsne_1[:, 1],
        label="20 Atom Test Set",
        s=2,
        color="lightgreen",
    )
    ax1.scatter(
        test_embeddings_with_TTA_tsne_1[:, 0],
        test_embeddings_with_TTA_tsne_1[:, 1],
        label="20 Atom Test Set with TTA",
        s=2,
        color="darkgreen",
    )
    ax1.scatter(
        test_embeddings_tsne_2[:, 0],
        test_embeddings_tsne_2[:, 1],
        label="NH2 Test Set",
        s=2,
        color="lightcoral",
    )
    ax1.scatter(
        test_embeddings_with_TTA_tsne_2[:, 0],
        test_embeddings_with_TTA_tsne_2[:, 1],
        label="NH2 Test Set with TTA",
        s=2,
        color="darkred",
    )
    ax1.scatter(
        val_embeddings_tsne[:, 0],
        val_embeddings_tsne[:, 1],
        label="Validation Set",
        s=2,
        color="peachpuff",
    )
    ax1.scatter(
        val_embeddings_with_TTA_tsne[:, 0],
        val_embeddings_with_TTA_tsne[:, 1],
        label="Validation Set with TTA",
        s=2,
        color="darkorange",
    )
    ax1.scatter(
        test_centroid_tsne_1[0],
        test_centroid_tsne_1[1],
        s=200,
        c="lightgreen",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="20 Atom Test Centroid",
    )
    ax1.scatter(
        test_tta_centroid_tsne_1[0],
        test_tta_centroid_tsne_1[1],
        s=200,
        c="darkgreen",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="20 Atom Test Centroid with TTA",
    )
    ax1.scatter(
        test_centroid_tsne_2[0],
        test_centroid_tsne_2[1],
        s=200,
        c="lightcoral",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="NH2 Test Centroid",
    )
    ax1.scatter(
        test_tta_centroid_tsne_2[0],
        test_tta_centroid_tsne_2[1],
        s=200,
        c="darkred",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="NH2 Test Centroid with TTA",
    )
    ax1.scatter(
        val_centroid_tsne[0],
        val_centroid_tsne[1],
        s=200,
        c="peachpuff",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid",
    )
    ax1.scatter(
        val_tta_centroid_tsne[0],
        val_tta_centroid_tsne[1],
        s=200,
        c="darkorange",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid with TTA",
    )
    ax1.scatter(
        train_centroid_tsne[0],
        train_centroid_tsne[1],
        s=200,
        c="blue",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Train Centroid",
    )

    ax1.set_title("t-SNE Projection")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    x_min_tsne, x_max_tsne, y_min_tsne, y_max_tsne = get_dynamic_limits(
        train_embeddings_tsne,
        val_embeddings_tsne,
        test_embeddings_tsne_1,
        test_embeddings_tsne_2,
        20,
        80,
    )

    ax1.set_xlim(x_min_tsne, x_max_tsne)
    ax1.set_ylim(y_min_tsne, y_max_tsne)
    fig.legend(bbox_to_anchor=(1, -0.3), loc="lower right", ncol=3)

    # UMAP plot on the right
    ax2.scatter(
        test_embeddings_umap_1[:, 0],
        test_embeddings_umap_1[:, 1],
        label="20 Atom Test Set",
        s=2,
        color="lightgreen",
    )
    ax2.scatter(
        test_embeddings_with_TTA_umap_1[:, 0],
        test_embeddings_with_TTA_umap_1[:, 1],
        label="20 Atom Test Set with TTA",
        s=2,
        color="darkgreen",
    )
    ax2.scatter(
        test_embeddings_umap_2[:, 0],
        test_embeddings_umap_2[:, 1],
        label="NH2 Test Set",
        s=2,
        color="lightcoral",
    )
    ax2.scatter(
        test_embeddings_with_TTA_umap_2[:, 0],
        test_embeddings_with_TTA_umap_2[:, 1],
        label="NH2 Test Set with TTA",
        s=2,
        color="darkred",
    )
    ax2.scatter(
        val_embeddings_umap[:, 0],
        val_embeddings_umap[:, 1],
        label="Validation Set",
        s=2,
        color="peachpuff",
    )
    ax2.scatter(
        val_embeddings_with_TTA_umap[:, 0],
        val_embeddings_with_TTA_umap[:, 1],
        label="Validation Set with TTA",
        s=2,
        color="darkorange",
    )

    ax2.scatter(
        test_centroid_umap_1[0],
        test_centroid_umap_1[1],
        s=200,
        c="lightgreen",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="20 Atom Test Centroid",
        alpha=0.9,
    )
    ax2.scatter(
        test_tta_centroid_umap_1[0],
        test_tta_centroid_umap_1[1],
        s=200,
        c="darkgreen",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="20 Atom Test Centroid with TTA",
        alpha=0.9,
    )
    ax2.scatter(
        test_centroid_umap_2[0],
        test_centroid_umap_2[1],
        s=200,
        c="lightcoral",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="NH2 Test Centroid",
        alpha=0.9,
    )
    ax2.scatter(
        test_tta_centroid_umap_2[0],
        test_tta_centroid_umap_2[1],
        s=200,
        c="darkred",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="NH2 Test Centroid with TTA",
        alpha=0.9,
    )
    ax2.scatter(
        val_centroid_umap[0],
        val_centroid_umap[1],
        s=200,
        c="peachpuff",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid",
        alpha=0.9,
    )
    ax2.scatter(
        val_tta_centroid_umap[0],
        val_tta_centroid_umap[1],
        s=200,
        c="darkorange",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid with TTA",
        alpha=0.9,
    )
    ax2.scatter(
        train_centroid_umap[0],
        train_centroid_umap[1],
        s=200,
        c="blue",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Train Centroid",
        alpha=0.9,
    )

    ax2.set_title("UMAP Projection")
    ax2.set_xlabel("UMAP Component 1")
    ax2.set_ylabel("UMAP Component 2")

    x_min_umap, x_max_umap, y_min_umap, y_max_umap = get_dynamic_limits(
        train_embeddings_umap,
        val_embeddings_umap,
        test_embeddings_umap_1,
        test_embeddings_umap_2,
        20,
        80,
    )

    ax2.set_xlim(x_min_umap, x_max_umap)
    ax2.set_ylim(y_min_umap, y_max_umap)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/sets_TTA.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/sets_TTA.pdf", bbox_inches="tight")
    plt.show()


@beartype
def loss_plot(
    combined_train_loss_list: list,
    denoise_train_loss_list: list,
    pred_train_loss_list: list,
    combined_val_loss_list: list,
    denoise_val_loss_list: list,
    pred_val_loss_list: list,
    save_plots: bool = False,
):
    """Plotting normalized losses for training and validation sets.
    Args:
        combined_train_loss_list (list): List of combined training losses.
        denoise_train_loss_list (list): List of denoise training losses.
        pred_train_loss_list (list): List of prediction training losses.
        combined_val_loss_list (list): List of combined validation losses.
        denoise_val_loss_list (list): List of denoise validation losses.
        pred_val_loss_list (list): List of prediction validation losses.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    @beartype
    def normalize_losses(loss_list: list) -> list:
        """Normalize a list of losses to the range [0, 1].
        Args:
            loss_list (list): List of losses to normalize.
        Returns:
            list: Normalized list of losses.
        """
        min_val, max_val = min(loss_list), max(loss_list)
        return [(loss - min_val) / (max_val - min_val) for loss in loss_list]

    combined_train_loss_list_norm = normalize_losses(combined_train_loss_list)
    denoise_train_loss_list_norm = normalize_losses(denoise_train_loss_list)
    pred_train_loss_list_norm = normalize_losses(pred_train_loss_list)
    combined_val_loss_list_norm = normalize_losses(combined_val_loss_list)
    denoise_val_loss_list_norm = normalize_losses(denoise_val_loss_list)
    pred_val_loss_list_norm = normalize_losses(pred_val_loss_list)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    epochs = len(combined_train_loss_list)

    # Plot normalized losses
    ax.plot(
        list(range(epochs)),
        combined_train_loss_list_norm,
        label="Combined Train Loss",
        color="blue",
    )
    ax.plot(
        list(range(epochs)),
        denoise_train_loss_list_norm,
        label="Denoise Train Loss",
        color="orange",
    )
    ax.plot(
        list(range(epochs)),
        pred_train_loss_list_norm,
        label="Pred Train Loss",
        color="green",
    )
    ax.plot(
        list(range(epochs)),
        combined_val_loss_list_norm,
        label="Combined Val Loss",
        color="blue",
        linestyle="dashed",
    )
    ax.plot(
        list(range(epochs)),
        denoise_val_loss_list_norm,
        label="Denoise Val Loss",
        color="orange",
        linestyle="dashed",
    )
    ax.plot(
        list(range(epochs)),
        pred_val_loss_list_norm,
        label="Pred Val Loss",
        color="green",
        linestyle="dashed",
    )

    ax.legend(ncol=2)
    ax.set_xlabel("Epochs")
    ax.set_xticks(list(range(0, epochs, 10)))
    ax.set_ylabel("Normalized Loss")

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/loss_plot.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/loss_plot.pdf", bbox_inches="tight")
    plt.show()


def feature_correlation(
    train_loader, val_loader, test_loader_20, test_loader_nh, feature_type: str, save_plots: bool = False
):
    """Plotting a correlation matrix for either node or edge features.
    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader_20: DataLoader for the 20 atom test set.
        test_loader_nh: DataLoader for the NH2 test set.
        feature_type (str): Type of features to analyze, either "node" or "edge".
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """
    assert feature_type in ["node", "edge"], "Choose either node or edge features."

    mean_features = []
    targets = []

    for loader in [train_loader, val_loader, test_loader_20, test_loader_nh]:
        for batch in loader:
            if feature_type == "node":
                features = batch.x
                index = batch.batch
            elif feature_type == "edge":
                features = batch.edge_attr
                index = batch.batch[batch.edge_index[0]]

            # Getting the mean value of each node feature over all nodes for each graph
            # For each sample in the batch, so mean_features_batch has shape (num_samples_in_batch, num_node_features)
            mean_features_batch = scatter_mean(features, index, dim=0).cpu().tolist()
            mean_features.extend(mean_features_batch)
            targets.extend(batch.y.cpu().tolist())

    mean_features = np.array(mean_features)
    targets = np.array(targets).reshape(-1, 1)

    # Concatenate features and target
    all_data = np.hstack([mean_features, targets])

    if feature_type == "node":
        feature_names = [
            "is_B",
            "is_Be",
            "is_Br",
            "is_C",
            "is_Cl",
            "is_F",
            "is_I",
            "is_N",
            "is_Nb",
            "is_O",
            "is_P",
            "is_S",
            "is_Se",
            "is_Si",
            "is_V",
            "is_W",
            "is_unknown_element",
            "has_degree_0",
            "has_degree_1",
            "has_degree_2",
            "has_degree_3",
            "has_degree_4",
            "has_degree_5",
            "has_unknown_degree",
            "formal_charge_-1",
            "formal_charge_-2",
            "formal_charge_1",
            "formal_charge_2",
            "formal_charge_0",
            "formal_charge_unknown",
            "num_H_0",
            "num_H_1",
            "num_H_2",
            "num_H_3",
            "num_H_4",
            "num_H_unknown",
            "hybridization_SP",
            "hybridization_SP2",
            "hybridization_SP3",
            "hybridization_SP3D",
            "hybridization_SP3D2",
            "hybridization_unknown",
            "is_aromatic",
            "mass",
        ]
    elif feature_type == "edge":
        feature_names = [
            "is_not_None",
            "is_single",
            "is_double",
            "is_triple",
            "is_aromatic",
            "is_conjugated",
            "is_ring",
        ]

    columns = feature_names + ["solubility"]

    # Create DataFrame and compute correlation matrix
    df = pd.DataFrame(all_data, columns=columns)
    corr_matrix = df.corr()

    if feature_type == "node":
        figsize = (11, 11)
    elif feature_type == "edge":
        figsize = (3, 3)

    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.index,
        cbar=False,
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if save_plots:
        plt.savefig(f"figures/feature_correlation_{feature_type}.jpg", dpi=300, bbox_inches="tight")
        plt.savefig(f"figures/feature_correlation_{feature_type}.pdf", bbox_inches="tight")


def feature_visualization(
    model,
    train_loader,
    val_loader,
    test_loader_20,
    test_loader_nh,
    feature_type: str,
    save_plots: bool = False,
):
    """
    Visualizes the original, noisy, and denoised node features of the first data point
    from the validation and test loaders.
    Args:
        model: The trained model to use for denoising.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader_20: DataLoader for the test set with 20+ atoms.
        test_loader_nh: DataLoader for the test set with NH2 functional groups.
        feature_type (str): Type of features to visualize, either "node" or "edge".
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """
    assert feature_type in ["node", "edge"], "Choose either node or edge features."

    model.eval()
    model.set_mode("denoise")

    loaders = [train_loader, val_loader, test_loader_20, test_loader_nh]
    y_dims = [loader.dataset.get(0).x.shape[0] for loader in loaders]
    # Subplot sizes are adapted to the number of features
    heights = [y / sum(y_dims) for y in y_dims]

    fig, axes = plt.subplots(
        len(loaders),
        3,
        figsize=(8, 10),
        gridspec_kw={"height_ratios": heights, "hspace": 0.15, "wspace": 0.05},
    )

    im_opts = {"cmap": "viridis", "vmin": 0, "vmax": 1, "aspect": "auto"}
    row_titles = ["Train", "Val", "Test 20", "Test NH2"]

    for i, loader in enumerate(loaders):
        data_point = loader.dataset.get(0)
        with torch.no_grad():
            # Getting the denoised features
            temp_loader = DataLoader([data_point.to(device)], batch_size=1)
            batch_for_model = next(iter(temp_loader))
            node_hat, edge_hat = model(batch_for_model)

        data_point = data_point.cpu()

        if feature_type == "node":
            arrays = [data_point.x, data_point.x_noisy, node_hat.cpu()]
        elif feature_type == "edge":
            arrays = [data_point.edge_attr, data_point.edge_attr_noisy, edge_hat.cpu()]

        for j, arr in enumerate(arrays):
            ax = axes[i, j]
            ax.imshow(arr, **im_opts)
            ax.set_ylim(arr.shape[0], 0)
            if j == 0:
                if feature_type == "node":
                    ax.set_ylabel(
                        f"{row_titles[i]}\n"
                        + ("Atoms" if feature_type == "node" else "Bonds")
                    )
                elif feature_type == "edge":
                    ax.set_ylabel("Bonds")
            if i == 0:
                ax.set_title(["Original", "Noisy", "Denoised"][j])
            if i == 2:
                ax.set_xlabel("Features")
            ax.set_xticks([])
            ax.set_yticks([])

    if save_plots:
        plt.savefig(f"figures/feature_visualization_{feature_type}.jpg", dpi=300, bbox_inches="tight")
        plt.savefig(f"figures/feature_visualization_{feature_type}.pdf", bbox_inches="tight")

    plt.show()
