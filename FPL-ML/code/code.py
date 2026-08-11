import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def dataset_preparation(keep_expected_stats):

    DATA_DIR = "data/"

    season_20_21 = pd.read_csv(f"{DATA_DIR}20-21.csv")
    season_21_22 = pd.read_csv(f"{DATA_DIR}21-22.csv")
    season_22_23 = pd.read_csv(f"{DATA_DIR}22-23.csv")
    season_23_24 = pd.read_csv(f"{DATA_DIR}23-24.csv")
    season_24_25 = pd.read_csv(f"{DATA_DIR}24-25.csv")

    season_20_21["season"] = "20-21"
    season_21_22["season"] = "21-22"
    season_22_23["season"] = "22-23"
    season_23_24["season"] = "23-24"
    season_24_25["season"] = "24-25"

    combined_data = pd.concat([season_20_21, season_21_22, season_22_23, season_23_24, season_24_25], ignore_index=True)
    combined_data = combined_data.drop(columns=[
        "mng_clean_sheets", "mng_draw", "mng_goals_scored",
        "mng_loss", "mng_underdog_draw", "mng_underdog_win",
        "mng_win", "modified", "transfers_in",
        "transfers_out", "transfers_balance", "selected",
        "kickoff_time", "element", "fixture",
        "xP", "ict_index"
    ], errors="ignore")

    # Expected statistics are always dropped from the dataset, however
    # this code was kept here to inform reviewers of how the project was
    # initially implemented, and to allow for reintroduction of expected statistics 
    # if desired in the future. The code below however is currently redundant as keep_expected_stats 
    # is always set to False in the main function
    expected_columns = [
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded"
    ]

    if not keep_expected_stats:
        combined_data = combined_data.drop(columns=expected_columns, errors="ignore")
    else:
        for column in expected_columns:
            if column in combined_data.columns:
                combined_data[column] = combined_data[column].fillna(0)
            else:
                combined_data[column] = 0

    return combined_data

def rolling_windows(dataset):
    dataset = dataset.sort_values(["name", "season", "round"]).reset_index(drop=True)

    window = 4
    numerical_stats = ["goals_scored", "assists", "saves", "own_goals",
                       "penalties_missed", "penalties_saved", "bonus", "bps",
                       "influence", "creativity", "threat", "minutes",
                       "goals_conceded"]
    binary_stats = ["clean_sheets", "yellow_cards", "red_cards"]

    for stats in numerical_stats:
        if stats in dataset.columns:
            dataset[f"{stats}_rolling_{window}"] = (
                dataset.groupby("name")[stats]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    for stats in binary_stats:
        if stats in dataset.columns:
            dataset[f"{stats}_rolling_{window}"] = (
                dataset.groupby("name")[stats]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    return dataset

def prepare_features(dataset):
    rolling_columns = [column for column in dataset.columns if "rolling" in column]
    fixture_info_columns = ["was_home", "opponent_team"]
    dataset["opponent_team"] = dataset["opponent_team"].astype("category").cat.codes
    dataset["was_home"] = dataset["was_home"].astype(int)

    features = rolling_columns + fixture_info_columns
    return dataset, features

def targets(dataset):
    dataset["clean_sheet_?"] = (dataset["clean_sheets"] >= 1).astype(int)
    dataset["yellow_card_?"] = (dataset["yellow_cards"] >= 1).astype(int)
    dataset["red_card_?"] = (dataset["red_cards"] >= 1).astype(int)
    return dataset

def fpl_point_conversion(predictions, position):
    points = 0

    if predictions["minutes"] >= 60:
        points += 2
    elif predictions["minutes"] > 0 and predictions["minutes"] < 60:
        points += 1
    
    goals = round(predictions["goals_scored"])
    if position == "GK":
        points += goals * 10
    elif position == "DEF":
        points += goals * 6
    elif position == "MID":
        points += goals * 5
    elif position == "FWD":
        points += goals * 4

    assists = round(predictions["assists"])
    points += assists * 3

    if predictions["clean_sheet_?"] == 1:
        if position in ["GK", "DEF"]:
            points += 4
        elif position == "MID":
            points += 1
    
    if position == "GK":
        saves = round(predictions["saves"])
        penalties_saved = round(predictions["penalties_saved"])
        points += saves // 3
        points += penalties_saved * 5
    
    penalties_missed = round(predictions["penalties_missed"])
    points -= penalties_missed * 2

    if position in ["GK", "DEF"]:
        goals_conceded = round(predictions["goals_conceded"])
        points -= (goals_conceded // 2) * 1

    yellow_cards = round(predictions["yellow_card_?"])
    red_cards = round(predictions["red_card_?"])
    points -= yellow_cards * 1
    points -= red_cards * 3

    own_goals = round(predictions["own_goals"])
    points -= own_goals * 2

    bonus_points = round(predictions["bonus"])
    points += bonus_points
    
    return points
        
def train_test_split(dataset):
    train_set = dataset[dataset["season"] != "24-25"]
    validation_set = dataset[(dataset["season"] == "24-25") & (dataset["round"] <= 28)]

    final_train_set = dataset[(dataset["season"] != "24-25") | (
        (dataset["season"] == "24-25") & (dataset["round"] <= 28))]

    test_set = dataset[(dataset["season"] == "24-25") & (dataset["round"] >= 29)]

    return train_set, validation_set, final_train_set, test_set

def weight_creation(dataset):
    decay_rate = 0.5
    season_order = {"20-21": 0, "21-22": 1, "22-23": 2, "23-24": 3, "24-25": 4}
    weights = dataset["season"].map(
        lambda season: decay_rate ** (4 - season_order[season])
    )
    return weights.values

def split_positions(position):
    if position == "GK":
        numerical_targets = ["goals_scored", "assists", "saves", "own_goals",
                             "penalties_missed", "penalties_saved", "minutes",
                             "goals_conceded", "bonus"]
    elif position == "DEF":
        numerical_targets = ["goals_scored", "assists", "own_goals",
                             "penalties_missed", "minutes",
                             "goals_conceded", "bonus"]
    elif position == "MID":
        numerical_targets = ["goals_scored", "assists", "own_goals",
                             "penalties_missed", "minutes", "bonus"]
    elif position == "FWD":
        numerical_targets = ["goals_scored", "assists", "own_goals",
                             "penalties_missed", "minutes", "bonus"]
        
    binary_targets = ["clean_sheet_?", "yellow_card_?", "red_card_?"]

    return numerical_targets, binary_targets

def random_forest(dataset, features, evaluate):
    print("-----------------RANDOM FOREST-----------------")

    all_results = []
    all_regressors = {}
    all_classifiers = {}

    for position in ["GK", "DEF", "MID", "FWD"]:
        print(f"Position: {position}")

        position_data = dataset[dataset["position"] == position].copy()
        train_set, validation_set, final_train_set, test_set = train_test_split(position_data)
        numerical_targets, binary_targets = split_positions(position)

        random_forest_regressor = {}
        random_forest_classifier = {}
        train_weights = weight_creation(train_set)
        final_train_weights = weight_creation(final_train_set)
        
        print("Random Forest Regressor")
        for target in numerical_targets:
            if target in train_set.columns:
                random_forest = RandomForestRegressor(n_estimators=100, random_state=453, n_jobs=-1)
                random_forest.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = random_forest.predict(validation_set[features])
                    validation_mae = mean_absolute_error(validation_set[target], validation_prediction)
                    validation_rsme = root_mean_squared_error(validation_set[target], validation_prediction)
                    print(f"{target}: Validation Mean Absolute Error: {validation_mae:.4f}, Validation Root Mean Squared Error: {validation_rsme:.4f}")

        print("Random Forest Classifier")
        for target in binary_targets:
            if target in train_set.columns:
                random_forest = RandomForestClassifier(n_estimators=100, random_state=453, n_jobs=-1, class_weight="balanced")
                random_forest.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = random_forest.predict(validation_set[features])
                    validation_accuracy = accuracy_score(validation_set[target], validation_prediction)
                    validation_f1_score = f1_score(validation_set[target], validation_prediction, zero_division=0)
                    print(f"{target}: Validation Accuracy: {validation_accuracy:.4f}, Validation F1 Score: {validation_f1_score:.4f}")

        print("Refitting in train + validation set")

        for target in numerical_targets:
            if target in final_train_set.columns:
                random_forest = RandomForestRegressor(n_estimators=100, random_state=453, n_jobs=-1)
                random_forest.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                random_forest_regressor[target] = random_forest

        for target in binary_targets:
            if target in final_train_set.columns:
                random_forest = RandomForestClassifier(n_estimators=100, random_state=453, n_jobs=-1, class_weight="balanced")
                random_forest.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                random_forest_classifier[target] = random_forest

        all_regressors[position] = random_forest_regressor
        all_classifiers[position] = random_forest_classifier

        if evaluate:
            print("Predicting test set")

            test_set = test_set.copy()
            all_predictions = {}

            for target in numerical_targets:
                if target in random_forest_regressor:
                    all_predictions[target] = random_forest_regressor[target].predict(test_set[features])

            for target in binary_targets:
                if target in random_forest_classifier:
                    all_predictions[target] = random_forest_classifier[target].predict(test_set[features])

            predicted_points = []
            for i in range(len(test_set)):
                predictions = {stat: all_predictions[stat][i] for stat in all_predictions}
                predicted_points.append(fpl_point_conversion(predictions, position))
            
            test_set["random_forest_predicted_points"] = predicted_points
            all_results.append(test_set)
            
    if evaluate:
        combined_test = pd.concat(all_results, ignore_index=True)
        mean_absolute_error_value, root_mean_squared_error_value = evaluate_model(combined_test, "random_forest_predicted_points", "Random Forest", dataset)

    if evaluate:
        return all_regressors, all_classifiers, combined_test, mean_absolute_error_value, root_mean_squared_error_value
    else:
        return all_regressors, all_classifiers, None, None, None

def xgboost(dataset, features, evaluate):
    print("-----------------XGBOOST-----------------")

    all_results = []
    all_regressors = {}
    all_classifiers = {}
    
    for position in ["GK", "DEF", "MID", "FWD"]:
        print(f"Position: {position}")

        position_data = dataset[dataset["position"] == position].copy()
        train_set, validation_set, final_train_set, test_set = train_test_split(position_data)
        numerical_targets, binary_targets = split_positions(position)

        xgboost_regressor = {}
        xgboost_classifier = {}
        train_weights = weight_creation(train_set)
        final_train_weights = weight_creation(final_train_set)

        print("XGBoost Regressor")
        for target in numerical_targets:
            if target in train_set.columns:
                xgboost = XGBRegressor(n_estimators=100, random_state=453, n_jobs=-1, verbosity=0)
                xgboost.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = xgboost.predict(validation_set[features])
                    validation_mae = mean_absolute_error(validation_set[target], validation_prediction)
                    validation_rsme = root_mean_squared_error(validation_set[target], validation_prediction)
                    print(f"{target}: Validation Mean Absolute Error: {validation_mae:.4f}, Validation Root Mean Squared Error: {validation_rsme:.4f}")

        print("XGBoost Classifier")
        for target in binary_targets:
            if target in train_set.columns:
                xgboost = XGBClassifier(n_estimators=100, random_state=453, n_jobs=-1, verbosity=0, scale_pos_weight=3)
                xgboost.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = xgboost.predict(validation_set[features])
                    validation_accuracy = accuracy_score(validation_set[target], validation_prediction)
                    validation_f1_score = f1_score(validation_set[target], validation_prediction, zero_division=0)
                    print(f"{target}: Validation Accuracy: {validation_accuracy:.4f}, Validation F1 Score: {validation_f1_score:.4f}")

        print("Refitting in train + validation set")

        for target in numerical_targets:
            if target in final_train_set.columns:
                xgboost = XGBRegressor(n_estimators=100, random_state=453, n_jobs=-1, verbosity=0)
                xgboost.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                xgboost_regressor[target] = xgboost

        for target in binary_targets:
            if target in final_train_set.columns:
                xgboost = XGBClassifier(n_estimators=100, random_state=453, n_jobs=-1, verbosity=0, scale_pos_weight=3)
                xgboost.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                xgboost_classifier[target] = xgboost

        all_regressors[position] = xgboost_regressor
        all_classifiers[position] = xgboost_classifier

        if evaluate:
            print("Predicting test set")

            test_set = test_set.copy()
            all_predictions = {}

            for target in numerical_targets:
                if target in xgboost_regressor:
                    all_predictions[target] = xgboost_regressor[target].predict(test_set[features])

            for target in binary_targets:
                if target in xgboost_classifier:
                    all_predictions[target] = xgboost_classifier[target].predict(test_set[features])

            predicted_points = []
            for i in range(len(test_set)):
                predictions = {stat: all_predictions[stat][i] for stat in all_predictions}
                predicted_points.append(fpl_point_conversion(predictions, position))
            
            test_set["xgboost_predicted_points"] = predicted_points
            all_results.append(test_set)

    if evaluate:
        combined_test = pd.concat(all_results, ignore_index=True)
        mean_absolute_error_value, root_mean_squared_error_value = evaluate_model(combined_test, "xgboost_predicted_points", "XGBoost", dataset)

    if evaluate:
        return all_regressors, all_classifiers, combined_test, mean_absolute_error_value, root_mean_squared_error_value
    else:
        return all_regressors, all_classifiers, None, None, None

def hybrid(dataset, features, evaluate):
    print("-----------------HYBRID MODEL (Poisson + Logistic)-----------------")

    all_results = []
    all_regressors = {}
    all_classifiers = {}

    for position in ["GK", "DEF", "MID", "FWD"]:
        print(f"Position: {position}")

        position_data = dataset[dataset["position"] == position].copy()
        train_set, validation_set, final_train_set, test_set = train_test_split(position_data)
        numerical_targets, binary_targets = split_positions(position)

        poisson_regressor = {}
        logistic_classifier = {}
        train_weights = weight_creation(train_set)
        final_train_weights = weight_creation(final_train_set)

        print("Poisson Regressor")
        for target in numerical_targets:
            if target in train_set.columns:
                poisson = PoissonRegressor(max_iter=5000)
                poisson.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = poisson.predict(validation_set[features])
                    validation_mae = mean_absolute_error(validation_set[target], validation_prediction)
                    validation_rsme = root_mean_squared_error(validation_set[target], validation_prediction)
                    print(f"{target}: Validation Mean Absolute Error: {validation_mae:.4f}, Validation Root Mean Squared Error: {validation_rsme:.4f}")

        print("Logistic Classifier")
        for target in binary_targets:
            if target in train_set.columns:
                logistic = LogisticRegression(max_iter=5000, class_weight="balanced")
                logistic.fit(train_set[features], train_set[target], sample_weight=train_weights)
                if evaluate:
                    validation_prediction = logistic.predict(validation_set[features])
                    validation_accuracy = accuracy_score(validation_set[target], validation_prediction)
                    validation_f1_score = f1_score(validation_set[target], validation_prediction, zero_division=0)
                    print(f"{target}: Validation Accuracy: {validation_accuracy:.4f}, Validation F1 Score: {validation_f1_score:.4f}")

        print("Refitting in train + validation set")

        for target in numerical_targets:
            if target in final_train_set.columns:
                poisson = PoissonRegressor(max_iter=5000)
                poisson.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                poisson_regressor[target] = poisson

        for target in binary_targets:
            if target in final_train_set.columns:
                logistic = LogisticRegression(max_iter=5000, class_weight="balanced")
                logistic.fit(final_train_set[features], final_train_set[target], sample_weight=final_train_weights)
                logistic_classifier[target] = logistic

        all_regressors[position] = poisson_regressor
        all_classifiers[position] = logistic_classifier

        if evaluate:
            print("Predicting test set")

            test_set = test_set.copy()
            all_predictions = {}

            for target in numerical_targets:
                if target in poisson_regressor:
                    all_predictions[target] = poisson_regressor[target].predict(test_set[features])

            for target in binary_targets:
                if target in logistic_classifier:
                    all_predictions[target] = logistic_classifier[target].predict(test_set[features])
        
            predicted_points = []
            for i in range(len(test_set)):
                predictions = {stat: all_predictions[stat][i] for stat in all_predictions}
                predicted_points.append(fpl_point_conversion(predictions, position))
            
            test_set["hybrid_predicted_points"] = predicted_points
            all_results.append(test_set)

    if evaluate:
        combined_test = pd.concat(all_results, ignore_index=True)
        mean_absolute_error_value, root_mean_squared_error_value = evaluate_model(combined_test, "hybrid_predicted_points", "Hybrid", dataset)

    if evaluate:
        return all_regressors, all_classifiers, combined_test, mean_absolute_error_value, root_mean_squared_error_value
    else:
        return all_regressors, all_classifiers, None, None, None
    
def player_prediction(dataset, features, random_forest_regressor, random_forest_classifier, xgboost_regressor, xgboost_classifier, hybrid_regressor, hybrid_classifier):

    while True:
        choice = input("Would you like to predict points for a player? (y/n): ")
        if choice == "n".lower():
            break
        elif choice != "y".lower():
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            continue

        teams = sorted(dataset[dataset["season"] == "24-25"]["team"].unique())
        print("Available teams for prediction:")
        count = 1
        for team in teams:
            print(f"{count}. {team}")
            count += 1

        while True:
            try:
                team_number = int(input("Enter team number: "))
                selected_team = teams[team_number - 1]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid team number.")
        
        players = dataset[(dataset["season"] == "24-25") & 
                        (dataset["team"] == selected_team) &
                        (dataset["position"].isin(["GK", "DEF", "MID", "FWD"]))
                        ][["name", "position"]].drop_duplicates().sort_values("name")
        
        print(f"Available players for {selected_team}:")

        count = 1
        player_list = players.values.tolist()
        for player in player_list:
            print(f"{count}. {player[0]} ({player[1]})")
            count += 1
        
        while True:
            try:
                player_choice = int(input("Enter player number: "))
                selected_player, player_position = player_list[player_choice - 1]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid player number.")
        
        available_gameweeks = sorted(dataset[(dataset["name"] == selected_player) &
                                    (dataset["season"] == "24-25") & 
                                    (dataset["round"] >= 29)]["round"].unique())
        
        if not available_gameweeks:
            print(f"{selected_player} did not feature in the testing gameweeks (GW 29-38), therefore no prediction can be made.")
            continue
        
        print(f"Available gameweeks for {selected_player} in the 24-25 season: {available_gameweeks}")

        while True:
            try:
                gameweek_choice = int(input("Enter gameweek number for prediction: "))
                if gameweek_choice in available_gameweeks:
                    break
                else:
                    print("Enter a valid gameweek number from the available options.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid gameweek number.")

        player_data = dataset[(dataset["name"] == selected_player) &
                            (dataset["season"] == "24-25") & 
                            (dataset["round"] == gameweek_choice)].sort_values("round")
                
        player_features = pd.DataFrame([player_data.iloc[-1][features]], columns=features)

        print(f"Predicting points for {selected_player}")

        model_predictions = {"Random Forest": None, "XGBoost": None, "Hybrid": None}

        for model_name, regressors, classifiers in [
            ("Random Forest", random_forest_regressor, random_forest_classifier),
            ("XGBoost", xgboost_regressor, xgboost_classifier),
            ("Hybrid", hybrid_regressor, hybrid_classifier)]:
            
            numerical_targets, binary_targets = split_positions(player_position)
            all_predictions = {}

            for target in numerical_targets:
                if target in regressors[player_position]:
                    all_predictions[target] = regressors[player_position][target].predict(player_features)[0]
            
            for target in binary_targets:
                if target in classifiers[player_position]:
                    all_predictions[target] = classifiers[player_position][target].predict(player_features)[0]

            predicted_points = fpl_point_conversion(all_predictions, player_position)
            print(f"{model_name}: {predicted_points} points")
            model_predictions[model_name] = predicted_points
        
        actual_points = player_data.iloc[-1]["total_points"]
        print(f"Actual points in GW{gameweek_choice}: {actual_points}")

def evaluate_model(test, predicted_column, name, dataset):
    total_minutes = dataset[dataset["season"] == "24-25"].groupby("name")["minutes"].sum()
    evaluated_players = total_minutes[total_minutes >= 90].index
    test = test[test["name"].isin(evaluated_players)]

    actual_points = test["total_points"].values
    predicted_points = test[predicted_column].values

    mean_absolute_error_value = mean_absolute_error(actual_points, predicted_points)
    root_mean_squared_error_value = root_mean_squared_error(actual_points, predicted_points)

    print(f"Model: {name}")
    print(f"Mean Absolute Error: {mean_absolute_error_value:.4f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error_value:.4f}")

    return mean_absolute_error_value, root_mean_squared_error_value

def visualisation(rf_test, xgb_test, hybrid_test, rf_mae, rf_rmse, xgb_mae, xgb_rmse, hybrid_mae, hybrid_rmse, dataset):
    season_minutes = dataset[dataset["season"] == "24-25"].groupby("name")["minutes"].sum()
    evaluated_players = season_minutes[season_minutes >= 90].index
    rf_test = rf_test[rf_test["name"].isin(evaluated_players)]
    xgb_test = xgb_test[xgb_test["name"].isin(evaluated_players)]
    hybrid_test = hybrid_test[hybrid_test["name"].isin(evaluated_players)]
    
    sns.set_theme(style="whitegrid")

    models = ["Random Forest", "XGBoost", "Hybrid"]
    mae_values = [rf_mae, xgb_mae, hybrid_mae]
    rmse_values = [rf_rmse, xgb_rmse, hybrid_rmse]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
    sns.barplot(x=models, y=mae_values, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Mean Absolute Error")
    axes[0].set_ylabel("MAE")
    axes[0].set_xlabel("Model")
    for i, value in enumerate(mae_values):
        axes[0].text(i, value + 0.01, f"{value:.4f}", ha="center", fontsize=10)

    sns.barplot(x=models, y=rmse_values, ax=axes[1], palette="Reds_d")
    axes[1].set_title("Root Mean Squared Error")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Model")
    for i, value in enumerate(rmse_values):
        axes[1].text(i, value + 0.01, f"{value:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Predicted vs Actual Points", fontsize=16, fontweight="bold")

    for ax, test, model_name, predicted_column in zip(
        axes,
        [rf_test, xgb_test, hybrid_test],
        models,
        ["random_forest_predicted_points", "xgboost_predicted_points", "hybrid_predicted_points"]
    ):
        ax.scatter(test["total_points"], test[predicted_column], alpha=0.3)
        max_val = max(test["total_points"].max(), test[predicted_column].max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
        ax.set_title(model_name)
        ax.set_xlabel("Actual Points")
        ax.set_ylabel("Predicted Points")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Actual vs Predicted Points Distribution", fontsize=16, fontweight="bold")

    for ax, test, model_name, predicted_column in zip(
        axes,
        [rf_test, xgb_test, hybrid_test],
        models,
        ["random_forest_predicted_points", "xgboost_predicted_points", "hybrid_predicted_points"]
    ):
        sns.kdeplot(test["total_points"], ax=ax, label="Actual Points", color="blue", linewidth=2)
        sns.kdeplot(test[predicted_column], ax=ax, label="Predicted Points", color="red", linewidth=2)
        ax.set_title(model_name)
        ax.set_xlabel("Points")
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MAE by Position", fontsize=16, fontweight="bold")

    positions = ["GK", "DEF", "MID", "FWD"]
    for ax, test, model_name, predicted_column in zip(
        axes,
        [rf_test, xgb_test, hybrid_test],
        models,
        ["random_forest_predicted_points", "xgboost_predicted_points", "hybrid_predicted_points"]
    ):
        position_mae_array = []
        for position in positions:
            position_test = test[test["position"] == position]
            if len(position_test) > 0:
                position_mae = mean_absolute_error(position_test["total_points"], position_test[predicted_column])
                position_mae_array.append(position_mae)
            else:
                position_mae_array.append(0)
        sns.barplot(x=positions, y=position_mae_array, ax=ax, palette="Blues_d")
        ax.set_title(model_name)
        ax.set_xlabel("Position")
        ax.set_ylabel("MAE")
        for i, value in enumerate(position_mae_array):
            ax.text(i, value + 0.01, f"{value:.4f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
def main():
    dataset = dataset_preparation(False)
    dataset = rolling_windows(dataset)
    dataset = targets(dataset)
    dataset, features = prepare_features(dataset)
    dataset = dataset.dropna(subset=features)

    while True:
        mode = input("Enter '1': Evaluate models. Enter '2': Evaluate models with graphs. Enter '3': Predict points for a player only: ")
        if mode == "1":
            evaluate = True
            graphs = False
            break
        elif mode == "2":
            evaluate = True
            graphs = True
            break
        elif mode == "3":
            evaluate = False
            graphs = False
            break
        else:
            print("Invalid input. Please enter '1', '2', or '3'.")

    rf_regressor, rf_classifier, rf_test, rf_mae, rf_rmse = random_forest(dataset, features, evaluate)
    xgb_regressor, xgb_classifier, xgb_test, xgb_mae, xgb_rmse = xgboost(dataset, features, evaluate)
    hybrid_regressor, hybrid_classifier, hybrid_test, hybrid_mae, hybrid_rmse = hybrid(dataset, features, evaluate)

    if graphs:
        visualisation(rf_test, xgb_test, hybrid_test, rf_mae, rf_rmse, xgb_mae, xgb_rmse, hybrid_mae, hybrid_rmse, dataset)

    player_prediction(dataset, features, rf_regressor, rf_classifier, xgb_regressor, xgb_classifier, hybrid_regressor, hybrid_classifier)

main()