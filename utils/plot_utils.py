import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from beartype import beartype


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
    df_train: pd.DataFrame, df_test_1: pd.DataFrame, df_test_2: pd.DataFrame, save_plots: bool = False
):
    """Plotting a histogram of solubility values for training and test datasets.
    Args:
        df_train (pd.DataFrame): Training dataset containing 'ExperimentalLogS' column.
        df_test_1 (pd.DataFrame): Test dataset containing 'LogS' column.
        df_test_2 (pd.DataFrame): Another test dataset containing 'LogS' column.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    assert (
        "ExperimentalLogS" in df_train.columns
    ), "Training Data must contain 'ExperimentalLogS' column"
    assert "LogS" in df_test_1.columns, "Test Data must contain 'LogS' column"
    assert "LogS" in df_test_2.columns, "Test Data must contain 'LogS' column"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(
        df_train["ExperimentalLogS"],
        bins=100,
        alpha=0.7,
        label="Training Data",
        color="blue",
        density=False,
    )

    ax.hist(
        df_test_1["LogS"],
        bins=100,
        alpha=0.7,
        label="Ether Test Data",
        color="green",
        density=False,
    )

    ax.hist(
        df_test_2["LogS"],
        bins=100,
        alpha=0.7,
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
        plt.savefig("figures/solubility_histogram_shift.jpg", dpi=300, bbox_inches="tight")
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
    fig, axes = plt.subplots(2, 4, figsize=(8, 6), sharey="row")
    axes = axes.flatten()

    for i, fg in enumerate(functional_groups):
        ax = axes[i]

        # Get the range of values for proper binning
        max_val = max(df_train[fg].max(), df_test[fg].max())
        bins = np.arange(
            -0.5, max_val + 1.5, 1
        )  # Creates bins centered on integer values

        # Plot histograms for both datasets
        ax.hist(
            df_train[fg],
            bins=bins,
            alpha=0.7,
            label="Training Data (AqSolDBc)",
            color="blue",
            density=False,
        )

        ax.hist(
            df_test[fg],
            bins=bins,
            alpha=0.7,
            label="Test Data (OChemUnseen)",
            color="orange",
            density=False,
        )

        ax.set_xlabel(fg.replace("fr_", "").replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, max_val + 0.5)

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
def get_dynamic_limits(
    train_embeddings,
    test_embeddings_1,
    test_embeddings_2,
    lower_percentile: int = 5,
    upper_percentile: int = 95,
):
    """Calculate dynamic limits for plotting embeddings based on percentiles.
    Args:
        train_embeddings (np.ndarray): Embeddings for the training set.
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

    x_max = max(x_max_train, x_max_test_1, x_max_test_2)
    x_min = min(x_min_train, x_min_test_1, x_min_test_2)
    y_max = max(y_max_train, y_max_test_1, y_max_test_2)
    y_min = min(y_min_train, y_min_test_1, y_min_test_2)

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
        train_embeddings_tsne, test_embeddings_tsne_1, test_embeddings_tsne_2
    )
    ax1.set_xlim(x_min_tsne, x_max_tsne)
    ax1.set_ylim(y_min_tsne, y_max_tsne)

    x_min_umap, x_max_umap, y_min_umap, y_max_umap = get_dynamic_limits(
        train_embeddings_umap, test_embeddings_umap_1, test_embeddings_umap_2
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
        label="Ether Test Set",
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
        train_embeddings_tsne, test_embeddings_tsne_1, test_embeddings_tsne_2
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
        label="Ether Test Set",
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
        train_embeddings_umap, test_embeddings_umap_1, test_embeddings_umap_2
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
    test_embeddings_tsne_1: np.ndarray,
    test_embeddings_tsne_2: np.ndarray,
    test_embeddings_with_TTA_tsne_1: np.ndarray,
    test_embeddings_with_TTA_tsne_2: np.ndarray,
    train_embeddings_umap: np.ndarray,
    val_embeddings_umap: np.ndarray,
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
        test_embeddings_tsne_1 (np.ndarray): t-SNE embeddings for a test set.
        test_embeddings_tsne_2 (np.ndarray): t-SNE embeddings for another test set.
        test_embeddings_with_TTA_tsne_1 (np.ndarray): t-SNE embeddings for a test set with TTA.
        test_embeddings_with_TTA_tsne_2 (np.ndarray): t-SNE embeddings for another test set with TTA.
        train_embeddings_umap (np.ndarray): UMAP embeddings for the training set.
        val_embeddings_umap (np.ndarray): UMAP embeddings for the validation set.
        test_embeddings_umap_1 (np.ndarray): UMAP embeddings for a test set.
        test_embeddings_umap_2 (np.ndarray): UMAP embeddings for another test set.
        test_embeddings_with_TTA_umap_1 (np.ndarray): UMAP embeddings for a test set with TTA.
        test_embeddings_with_TTA_umap_2 (np.ndarray): UMAP embeddings for another test set with TTA.
        save_plots (bool): Whether to save the plots as images. Defaults to False.
    """

    # Calculate centroids with median instead of mean since we have outliers
    train_centroid_umap = np.median(train_embeddings_umap, axis=0)
    val_centroid_umap = np.median(val_embeddings_umap, axis=0)
    test_centroid_umap_1 = np.median(test_embeddings_umap_1, axis=0)
    test_tta_centroid_umap_1 = np.median(test_embeddings_with_TTA_umap_1, axis=0)
    test_centroid_umap_2 = np.median(test_embeddings_umap_2, axis=0)
    test_tta_centroid_umap_2 = np.median(test_embeddings_with_TTA_umap_2, axis=0)

    # Calculate centroids
    train_centroid_tsne = np.mean(train_embeddings_tsne, axis=0)
    val_centroid_tsne = np.mean(val_embeddings_tsne, axis=0)
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
        label="Ether Test Set",
        s=2,
        color="lightgreen",
    )
    ax1.scatter(
        test_embeddings_with_TTA_tsne_1[:, 0],
        test_embeddings_with_TTA_tsne_1[:, 1],
        label="Ether Test Set with TTA",
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
        test_centroid_tsne_1[0],
        test_centroid_tsne_1[1],
        s=200,
        c="lightgreen",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Ether Test Centroid",
    )
    ax1.scatter(
        test_tta_centroid_tsne_1[0],
        test_tta_centroid_tsne_1[1],
        s=200,
        c="darkgreen",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="Ether Test Centroid with TTA",
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
        train_centroid_tsne[0],
        train_centroid_tsne[1],
        s=200,
        c="blue",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Train Centroid",
    )
    ax1.scatter(
        val_centroid_tsne[0],
        val_centroid_tsne[1],
        s=200,
        c="orange",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid",
    )

    ax1.set_title("t-SNE Projection")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    x_min_tsne, x_max_tsne, y_min_tsne, y_max_tsne = get_dynamic_limits(
        train_embeddings_tsne, test_embeddings_tsne_1, test_embeddings_tsne_2, 20, 80
    )

    ax1.set_xlim(x_min_tsne, x_max_tsne)
    ax1.set_ylim(y_min_tsne, y_max_tsne)
    fig.legend(bbox_to_anchor=(0.95, -0.25), loc="lower right", ncol=3)

    # UMAP plot on the right
    ax2.scatter(
        test_embeddings_umap_1[:, 0],
        test_embeddings_umap_1[:, 1],
        label="Ether Test Set",
        s=2,
        color="lightgreen",
    )
    ax2.scatter(
        test_embeddings_with_TTA_umap_1[:, 0],
        test_embeddings_with_TTA_umap_1[:, 1],
        label="Ether Test Set with TTA",
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
        test_centroid_umap_1[0],
        test_centroid_umap_1[1],
        s=200,
        c="lightgreen",
        marker="P",
        edgecolors="black",
        linewidths=1,
        label="Ether Test Centroid",
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
        label="Ether Test Centroid with TTA",
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
    ax2.scatter(
        val_centroid_umap[0],
        val_centroid_umap[1],
        s=200,
        c="orange",
        marker="X",
        edgecolors="black",
        linewidths=1,
        label="Val Centroid",
        alpha=0.9,
    )

    ax2.set_title("UMAP Projection")
    ax2.set_xlabel("UMAP Component 1")
    ax2.set_ylabel("UMAP Component 2")

    x_min_umap, x_max_umap, y_min_umap, y_max_umap = get_dynamic_limits(
        train_embeddings_umap, test_embeddings_umap_1, test_embeddings_umap_2, 20, 80
    )

    ax2.set_xlim(x_min_umap, x_max_umap)
    ax2.set_ylim(y_min_umap, y_max_umap)

    plt.tight_layout()
    if save_plots:
        plt.savefig("figures/sets_TTA.jpg", dpi=300, bbox_inches="tight")
        plt.savefig("figures/sets_TTA.pdf", bbox_inches="tight")
    plt.show()
