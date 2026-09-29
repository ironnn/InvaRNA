import argparse
import pandas as pd
from pathlib import Path

from src.train import train_model_nested_cv
from src.transfer_learning import transfer_learning
from src.predict import predict_using_nested_cross_validation_models,predict_using_models_trained_in_one_fold
from src.utils.helpers import (
    load_config,
)




def main(args) -> None:
    """Main process"""

    if args.train:
        # Train 10*9 models to predict individual human TEs (78 samples),
        # with extremely long 5'UTR, CDS, and 3'UTR transcripts removed
        config = load_config("config/conf.yml")
        config["pad_5_prime"] = True
        config["target_column_pattern"] = "^TE_"
        config["optimizer"] = "AdamW"
        config["lr"] = 0.0001
        config["l2_scale"] = 0.001
        config["with_NAs"] = True
        config["max_shift"] = 0
        config["num_conv_layers"] = 10
        config["symmetric_shift"] = True
        config["remove_extreme_txs"] = True
        config["activation"] = "relu"
        config["max_utr5_len"] = 1_381  # used when training the human multi-task model
        config["max_cds_utr3_len"] = 11_937  # used when training the human multi-task model

        train_model_nested_cv(
            config,
            inner_cv_folds=9,
            experiment_name="human_multi-task_TE"
        )

        # Predict TEs from scratch without transfer learning
        config["pad_5_prime"] = True
        config["optimizer"] = "AdamW"
        config["lr"] = 0.0001
        config["l2_scale"] = 0.001
        config["with_NAs"] = False
        config["activation"] = "relu"
        config["max_shift"] = 0
        config["num_conv_layers"] = 10
        config["remove_extreme_txs"] = True
        config["max_utr5_len"] = 1_381  # used when training the human multi-task model
        config["max_cds_utr3_len"] = 11_937  # used when training the human multi-task model

        for target_name in (
            "TE_HeLa_S3",
            "TE_neurons",  
            "TE_H1-hESC",
            "TE_H9-hESC",
            "TE_HSPCs",
            "TE_Kidney_normal_tissue",
            "TE_muscle_tissue",
            "TE_HCC_tumor",
            "TE_HCC_adjancent_normal",
        ):
            config["target_column_pattern"] = target_name
            train_model_nested_cv(
                config,
                inner_cv_folds=9,
                experiment_name=target_name.replace("TE_", "").replace("-", "_") + "(from_scratch)"
            )

        # Predict TEs in all human samples without aligning sequences at the start codon
        config["pad_5_prime"] = False
        config["target_column_pattern"] = "^TE_"
        config["optimizer"] = "AdamW"
        config["lr"] = 0.0001
        config["l2_scale"] = 0.001
        config["with_NAs"] = True
        config["max_shift"] = 0
        config["num_conv_layers"] = 10
        config["symmetric_shift"] = True
        config["remove_extreme_txs"] = True
        config["activation"] = "relu"
        config["max_utr5_len"] = 1_381  # used when training the human multi-task model
        config["max_cds_utr3_len"] = 11_937  # used when training the human multi-task model

        train_model_nested_cv(
            config,
            inner_cv_folds=9,
            experiment_name="human_multi-task_TE(without_aligning_start_codons)"
        )

        # Predict TEs in all human samples without labeling the first nt of each codon
        config["pad_5_prime"] = True
        config["target_column_pattern"] = "^TE_"
        config["optimizer"] = "AdamW"
        config["lr"] = 0.0001
        config["l2_scale"] = 0.001
        config["with_NAs"] = True
        config["max_shift"] = 0
        config["num_conv_layers"] = 10
        config["symmetric_shift"] = True
        config["remove_extreme_txs"] = True
        config["activation"] = "relu"
        config["label_codons"] = False
        config["max_utr5_len"] = 1_381  # used when training the human multi-task model
        config["max_cds_utr3_len"] = 11_937  # used when training the human multi-task model

        train_model_nested_cv(
            config,
            inner_cv_folds=9,
            experiment_name="human_multi-task_TE(without_labeling_codons)"
        )


    # Transfer learning using pretrained models
    if args.transfer_learning:
        pretrain_run_df = pd.read_csv("models/runs.csv")
        config = load_config("config/conf.yml")
        config["pad_5_prime"] = True
        config["with_NAs"] = False
        config["max_shift"] = 0
        config["num_conv_layers"] = 10
        config["symmetric_shift"] = True
        config["remove_extreme_txs"] = False
        config["activation"] = "relu"
        config["optimizer"] = "AdamW"
        config["l2_scale"] = 0.001
        config["max_utr5_len"] = 1_381  # used when training the human multi-task model
        config["max_cds_utr3_len"] = 11_937  # used when training the human multi-task model

        for target_name in (
            "TE_HCC_tumor", 
            "TE_HCC_adjancent_normal", 
            "TE_HMECs",
            "TE_OSCC",
            "TE_T47D",
            "TE_HeLa_S3",
            "TE_neurons",
            "TE_H1-hESC",
            "TE_H9-hESC",
            "TE_HSPCs",  
            "TE_Kidney_normal_tissue",  
            "TE_muscle_tissue",  
        ):
            config["target_column_pattern"] = target_name

            transfer_learning(
                config,
                pretrain_run_df,
                outer_cv_folds = 10,
                inner_cv_folds = 9,
                stop_after_n_inner_cv_folds = None,
                random_state = 42,
                phase_1_epochs=50,
                max_epochs=200,
                patience=50,
                experiment_name=target_name.replace("TE_", "").replace("-", "_")
            )


    if args.predict:
        # Predict TEs in multiple tissue/cell lines using pretrained nested
        # cross-validation models. Predictions by models trained using the 
        # same test fold (as indicated by the "fold" column) are averaged 
        # across the models. The averaged predictions by models using different
        # test folds are then concatenated.
        input_file = "data/prediction_input1.txt"
        # input_file = "data/prediction_input2.txt" # Alternative input file
        output_file = "results/prediction_output.txt"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        run_df = pd.read_csv("models/runs.csv")
        predictions = predict_using_nested_cross_validation_models(
            input_file, 
            run_df, 
            5,
            batch_size = 32, 
            num_workers=4) 
        
        # Calculate the mean tissue/cell-specific TE across the test folds
        columns_to_aggregate = [col for col in predictions.columns if col.startswith("predicted_")]
        predicted_TE = predictions.groupby(
            ["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"],
            as_index=False)[columns_to_aggregate].agg('mean')

        # Calculate the mean TE across tissue/cell types
        predicted_TE["mean_predicted_TE"] = predicted_TE[columns_to_aggregate].mean(axis=1)

        # Write predictions to disk
        predicted_TE.to_csv(output_file, index=False, sep="\t")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train", default=False, action="store_true", help="Train the multitask model"
    )

    parser.add_argument(
        "--transfer_learning",
        default=False,
        action="store_true",
        help="Fine-tune a pretrained model using transfer learning",
    )

    parser.add_argument(
        "--predict", default=False, action="store_true", help="Predictions"
    )

    main(parser.parse_args())
