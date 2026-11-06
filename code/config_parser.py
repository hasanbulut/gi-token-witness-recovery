import yaml
import itertools

class ExperimentConfig:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.settings = self.config["experiment_settings"]
        self.families = self.config["families"]
        self.attacks = self.config["attacks"]
        self.params = self.config["parameters"]
        self.f_specific = self.config["family_specific_parameters"]
        self.noise = self.config["noise"]

    def generate_base_grid(self):
        """
        Temel parametre kombinasyonlarını (n ve d) üretir.
        """
        n_vals = self.params["n"]
        d_vals = self.params["d"]
        return list(itertools.product(n_vals, d_vals))

    def get_timeouts(self):
        return self.params["timeouts_sec"]

if __name__ == "__main__":
    parser = ExperimentConfig()
    print(f"Hedeflenen Hücre Başına Örnek Sayısı: {parser.settings['instances_per_cell']}")
    print(f"Temel n x d Kombinasyonları: {parser.generate_base_grid()}")