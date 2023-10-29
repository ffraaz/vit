import os

import vit


def test_vit():
    model = vit.main()
    dataloader_val = vit.get_dataloader(
        data_path=os.path.join(vit.repository_root(), "data"),
        batch_size=32,
        train_set=False,
    )
    accuracy = vit.evaluate(model, eval_size=10000, dataloader=dataloader_val)
    assert accuracy >= 0.9
