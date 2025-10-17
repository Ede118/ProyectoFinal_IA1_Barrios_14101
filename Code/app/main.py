from Code.app.Controller import Controller
from Code.ui.UI import UI
# instanciá ImgOrchestrator, BayesAgent, KNNModel con tus configs reales

def main():
    # TODO: cargar modelos entrenados desde disco
    ctrl = Controller(vision=..., bayes=..., knn=..., ui=UI())
    ctrl.run_episode()

if __name__ == "__main__":
    main()
