
class UI:
    def show_count(self, conteo: dict[str,int]) -> None:
        print("Conteo en la muestra (10 piezas):")
        for k,v in conteo.items():
            print(f"  - {k}: {v}")

    def show_proportion(self, labels: list[str], post: list[float]) -> None:
        print("Posterior de hipótesis (caja):")
        for lbl,p in zip(labels, post):
            print(f"  - {lbl}: {p:.3f}")

    def notify(self, msg: str) -> None:
        print(msg)



