import torch

from src.models.baseline import (
    LSTMBaseline,
    create_baseline_model,
)


def test_model_creation():
    model = create_baseline_model()

    assert isinstance(
        model,
        LSTMBaseline,
    )

    print(
        "Model creation: PASS"
    )


def test_model_architecture():
    model = create_baseline_model()

    assert model.input_size == 21
    assert model.hidden_size == 128
    assert model.num_layers == 2
    assert model.dropout_rate == 0.30

    print(
        "Model architecture configuration: PASS"
    )


def test_forward_pass():
    model = create_baseline_model()

    # ---------------------------------------------------------
    # SensorGuard input:
    #
    # batch_size = 8
    # window_size = 30
    # sensor_count = 21
    # ---------------------------------------------------------

    X = torch.randn(
        8,
        30,
        21,
    )

    model.eval()

    with torch.no_grad():
        output = model(X)

    assert output.shape == (
        8,
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "Forward pass: PASS"
    )

    print(
        f"Input shape:  {tuple(X.shape)}"
    )

    print(
        f"Output shape: {tuple(output.shape)}"
    )


def test_loss_and_backward():
    model = create_baseline_model()

    X = torch.randn(
        8,
        30,
        21,
    )

    y = torch.randint(
        low=0,
        high=2,
        size=(8,),
    ).float()

    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()

    logits = model(X)

    loss = criterion(
        logits,
        y,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    gradients_exist = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradients_exist = True

    assert gradients_exist

    print(
        "Loss computation: PASS"
    )

    print(
        "Backward propagation: PASS"
    )

    print(
        f"Loss: {loss.item():.6f}"
    )


def test_probability_conversion():
    model = create_baseline_model()

    X = torch.randn(
        8,
        30,
        21,
    )

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        )

    assert probabilities.shape == (
        8,
    )

    assert torch.all(
        probabilities >= 0.0
    )

    assert torch.all(
        probabilities <= 1.0
    )

    print(
        "Probability conversion: PASS"
    )

    print(
        f"Probability range: "
        f"{probabilities.min().item():.4f} - "
        f"{probabilities.max().item():.4f}"
    )


def test_invalid_input_shape():
    model = create_baseline_model()

    invalid_input = torch.randn(
        8,
        21,
    )

    try:
        model(invalid_input)

    except ValueError:

        print(
            "Invalid input validation: PASS"
        )

        return

    raise AssertionError(
        "Model should reject a 2D input tensor."
    )


def test_invalid_sensor_count():
    model = create_baseline_model()

    invalid_input = torch.randn(
        8,
        30,
        20,
    )

    try:
        model(invalid_input)

    except ValueError:

        print(
            "Sensor-count validation: PASS"
        )

        return

    raise AssertionError(
        "Model should reject input with "
        "incorrect sensor count."
    )


if __name__ == "__main__":

    print("=" * 80)

    print(
        "SensorGuard - Baseline LSTM Tests"
    )

    print("=" * 80)

    test_model_creation()

    test_model_architecture()

    test_forward_pass()

    test_loss_and_backward()

    test_probability_conversion()

    test_invalid_input_shape()

    test_invalid_sensor_count()

    print("=" * 80)

    print(
        "ALL BASELINE MODEL TESTS PASSED"
    )

    print("=" * 80)
    