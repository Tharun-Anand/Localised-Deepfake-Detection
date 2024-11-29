import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import click
import os
import sys

from tabulate import tabulate
def pretty_print_table(video_paths, preds, probs):
    label_mapping = {0: "real", 1: "fake"}
    table_data = [
        {"Video": os.path.basename(path), "Label": label_mapping[label], "Probability": f"{prob[label]:.2f}"}
        for path, label, prob in zip(video_paths, preds, probs)
    ]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))

def no_output(): # hide jargon output
    import contextlib
    class SuppressOutput(contextlib.ContextDecorator):
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            self._null = open(os.devnull, 'w')
            sys.stdout = self._null
            sys.stderr = self._null
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._null.close()
    return SuppressOutput()


@click.command()
@click.option("--video_path", type=str, required=True,prompt="Enter the video path or folder",help="Enter the video path")
@click.option("--weights_path", type=str, required=False,help="Enter the weights path")
@click.option("--device", type=str, required=False, default=None,help="Enter the device")
@click.option("--batch_size", type=int, required=False, default=4,help="Enter the batch size")
def main(video_path, weights_path=None, device=None,batch_size=4):
    with no_output():
        from main.dataset import MyDataset
        from main.model import Model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    weights_path = weights_path or "./weights/weights_remapped.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval().to(device)
    print(f"Loaded weights from {weights_path}")

    videos = []
    if os.path.isdir(video_path):
        for file in os.listdir(video_path):
            if file.endswith(".mp4"):
                videos.append(os.path.join(video_path, file))
    else:
        videos.append(video_path)

    dataset = MyDataset(videos)
    val_dataloader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    probs = []
    for batch in tqdm(val_dataloader,total=len(val_dataloader)):
        video = batch['video'].to(device)
        video = video/255
        video = video.permute(0,4,1,2,3)
        with torch.no_grad():
            logits = model(video)
        _pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
        _probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        preds.extend(_pred)
        probs.extend(_probs)

    pretty_print_table(dataset.video_paths, preds, probs)
    # for path, label, prob in zip(dataset.video_paths, preds, probs):
        # video = os.path.basename(path)
        # print(f"{video} {label} with probabilities {prob}")

if __name__ == "__main__":
    main()



