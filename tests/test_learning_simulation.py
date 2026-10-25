from src.learning_simulation import generic_neuroanatomy_figure


def test_learning_scene_has_generic_layers_and_no_patient_claim():
    figure = generic_neuroanatomy_figure()
    assert len(figure.data) == 6
    assert "teaching model only" in figure.layout.title.text
