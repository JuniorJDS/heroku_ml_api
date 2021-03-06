import argparse
import logging

from src.basic_cleaning import clear
from src.train_model import execute_train_test_model
from src.check_model import execute_check_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    if args.step_name == "all_steps" or args.step_name == "clear_data":
        logging.info("Cleaning data ...")
        clear()
        logging.info("Dataset has been cleaned!")

    if args.step_name == "all_steps" or args.step_name == "train_test_model":
        logging.info("Training and testing model ...")
        execute_train_test_model()
        logger.info("Training and testing done!")

    if args.step_name == "all_steps" or args.step_name == "check_score":
        logging.info("Checking the Score...")
        execute_check_model()
        logging.info("Checking the Score done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ML Pipeline"
    )
    parser.add_argument(
        "--step_name",
        type=str,
        choices=[
            "clear_data",
            "train_test_model",
            "check_score",
            "all_steps"
        ],
        default="all_steps"
    )

    args = parser.parse_args()

    go(args)
