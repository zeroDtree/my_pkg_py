import torch
from torchviz import make_dot


def get_compute_graph(
    model: torch.nn.Module,
    input_shape: tuple | None = None,
    input: dict | None = None,
    dir: str = "compute_graph",
    filename: str = "simple_net_graph",
    format: str = "pdf",
) -> None:
    """Generate the computing graph of model (format default is pdf)

    Args:
        model (torch.nn.Module): the model to generate the computing graph
        input_shape (tuple, optional): the shape of the input. Defaults to None.
        input (dict, optional): the input of the model. Defaults to None.
        dir (str, optional): the directory to save the computing graph. Defaults to "compute_graph".
        filename (str, optional): the name of the file to save the computing graph. Defaults to "simple_net_graph".
        format (str, optional): the format of the computing graph. Defaults to "pdf".


    Returns:
        None
    """
    assert input is not None or input_shape is not None, "error: input is None and input_shape is None"
    if input is None:
        assert input_shape is not None
        out = model(torch.randn(input_shape))
    else:
        out = model(**input)

    def extract_tensors(output):
        if isinstance(output, torch.Tensor):
            return output
        elif hasattr(output, "logits"):
            return output.logits
        elif hasattr(output, "loss"):
            return output.loss
        else:
            raise ValueError("Unsupported output type")

    output = extract_tensors(out)
    make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    ).render(dir + "/" + filename, format=format)
