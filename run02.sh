dmm-map pca --config config/config_dkf01.json --dim 2
dmm-map umap --config config/config_dkf01.json --dim 2 --data_sampling
dmm-plot pca --config config/config_dkf01.json --input pca.jbl --x_plot_type heatmap --z_plot_type heatmap
dmm-plot umap --config config/config_dkf01.json --input umap.jbl --x_plot_type heatmap --z_plot_type heatmap
