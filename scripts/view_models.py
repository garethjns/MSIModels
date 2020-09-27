from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier


def report_model(model_type: str):
    mod = MultisensoryClassifier(model_type)
    mod.build_model()
    try:
        mod.plot_dag()
    except (AssertionError, ImportError):
        print(f'Draw failed.')

    print(f'Model: {model_type}')
    print(f'n params: {mod.n_params}')


if __name__ == "__main__":

    for mod_type in ["early_integration", "intermediate_integration", "late_integration"]:
        report_model(mod_type)
