import sys

from src.models import predict_model as pm


def has_valid_args(args):
    """Determine whether the arguments passed are valid or not.

    Keyword arguments:
    args -- arguments passed (program name skipped)
    """
    return len(args) == 4 and not has_wrongly_positioned_args(args)


def display_help():
    """Display a message indicating the command usage to run the prediction app entry point."""
    print('Usage: poetry run python .\\app.predict.py -m \"MODEL\" -f CSV_FILE')
    print('Try \'poetry run python .\\app.predict.py --help for help.\'')


def has_wrongly_positioned_args(args):
    """Determine whether the arguments passed are wrongly positioned.

    Keyword arguments:
    args -- arguments passed (program name skipped)
    """
    return args[0] != '-m' \
           or args[2] != '-f' \
           or has_wrong_model(args)


def has_wrong_model(args):
    """Determine whether the model name provided is wrong.

    Keyword arguments:
    args -- arguments passed (program name skipped)
    """
    return args[1] != 'gb' \
           and args[1] != 'xgb' \
           and args[1] != 'rf'


def get_model_prediction():
    """Get model prediction for given model and given CSV file path, otherwise display help message."""
    args = sys.argv[1:]
    if has_valid_args(args):
        model = args[1]
        file_path = args[3]
        pm.predict(model, file_path)
    else:
        display_help()


if __name__ == '__main__':
    get_model_prediction()