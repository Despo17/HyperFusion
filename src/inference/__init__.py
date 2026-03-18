def __init__(self, model_path=MODEL_PATH):

    # ✅ Download model if not exists
    download_model()

    self.model = build_multi_asset_hyperfusion(
        seq_len=30,
        num_features=10,
        num_assets=len(ASSETS)
    )

    self.model.load_weights(model_path)

    self.asset_to_id = {
        asset: i for i, asset in enumerate(ASSETS.keys())
    }