from torch.utils.data import DataLoader
import torch
from src.dataset import TreeHealthDataset
from src.data.etl_layer import ETL as etl
from src.api.schemas import InputBatch, OutputBatch
import sys

sys.path.append("...")


async def model_inference(input_batch: InputBatch) -> OutputBatch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_input = etl().data_preprocessing(input_batch)

    model = torch.load("models/torch_models/model_v1.pth", weights_only=False)
    model.to(device)
    model.eval()

    inference_dataset = TreeHealthDataset(df=processed_input)

    inference_loader = DataLoader(inference_dataset, batch_size=512, shuffle=False)
    model_predictions = model.predict(inference_loader, device)

    output_batch = etl().data_postprocessing(model_predictions, input_batch)

    return output_batch
