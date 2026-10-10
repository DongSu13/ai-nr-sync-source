from gpu_mlp_ablation import validate_feature_schema


def test_rejects_crc_and_truth_fields_from_gpu_mlp_features():
    records = [
        {
            "features": {
                "pss_metric": 0.9,
                "sss_metric": 0.8,
                "correlation_metric": 1.7,
                "abs_cfo_hz": 100.0,
                "received_power": 1.0,
                "residual_power": 0.1,
            },
            "label": 1,
        }
    ]
    validate_feature_schema(records)
