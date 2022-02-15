from typing import Dict
from pathlib import Path
import SimpleITK
import torch

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device
from algorithm.preprocess import preprocess
from algorithm.i3d.i3dpt import I3D
from scripts.infer_clstask import CovidSeverePredictor

COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        # load model
        self.model = I3D(nr_outputs=2)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load('./algorithm/model_covid.pth', map_location=torch.device(device)))
        self.model = self.model.eval()

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image = to_input_format(input_image)

        # run model
        with torch.no_grad():
            output = torch.sigmoid(self.model(input_image))
        prob_covid, prob_severe = unpack_single_output(output)

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


class StoicAlgorithmDJ(MultiClassAlgorithm):

    AGE_MAP = {35: 1, 45: 2, 55: 3, 65: 4, 75: 5, 85: 6}
    SEX_MAP = {'F': 0, 'M': 1, 'A': 2, 'O': 2, 'N': 2}
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )
        
        # load model
        # git_rt, cfg.model_name, cfg.best_weight, device = f'cuda:{cfg.gpu_ix}'
        model_dir = './work_dirs'
        model_name = 'resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode'
        best_weight = 'best_auc_epoch_6.pth'
        self.classifier = CovidSeverePredictor(model_dir = model_dir, 
                                            model_name = model_name, 
                                            best_weight= best_weight, 
                                            device = 'cuda:0')


    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        # input_image = preprocess(input_image)
        # input_image = to_input_format(input_image)
        case_info = {}
        for k in input_image.GetMetaDataKeys():
            case_info[k] = input_image.GetMetaData(k)

        age, sex = case_info.get('PatientAge', 1), case_info.get('PatientSex', 'N')
        age_num = self.AGE_MAP[age]
        
        # run model
        result_1x2 = self.classifier(input_image, age_num)
        prob_covid, prob_severe = result_1x2[0] 

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithmDJ().process()
