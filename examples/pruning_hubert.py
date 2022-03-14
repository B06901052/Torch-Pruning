import torch
import torch_pruning as tp
import s3prl.hub as hub

if __name__ == "__main__":
    device = "cpu"
    model = getattr(hub, "hubert")().to(device).eval()
    wavs = [torch.randn(160000, dtype=torch.float, device=device)]
    
    # 1. setup strategy (L1 Norm)
    strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

    # 2. build layer dependency for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, inputs=wavs, output_transform=lambda x: x["last_hidden_state"])

    # 3. get a pruning plan from the dependency graph.
    pruning_idxs = strategy(model.model.feature_extractor.conv_layers[0][0].weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9, ...]
    pruning_plan = DG.get_pruning_plan( model.model.feature_extractor.conv_layers[0][0], tp.prune_conv, idxs=pruning_idxs )
    print(pruning_plan)

    # 4. execute this plan (prune the model)
    pruning_plan.exec()