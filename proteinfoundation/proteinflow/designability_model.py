from fast_designability import Designability
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.coors_utils import nm_to_ang

class DesignabilityModel(Proteina):

    def __init__(self, cfg_exp):
        super().__init__(cfg_exp)

    def on_predict_start(self):
        self.designability = Designability(self.device)
        self.csv_fname = f"/dataset/pdb/designability/designability_pdb_{self.global_rank}.csv"

    def predict_step(self, batch, batch_idx):

        graph, filenames = batch
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(graph)
        scores = self.designability.scRMSD(nm_to_ang(x_1))
        with open(self.csv_fname, "a") as f:
            for i in range(x_1.shape[0]):
                f.write(f"{filenames[i]},{scores[i].item()}\n")