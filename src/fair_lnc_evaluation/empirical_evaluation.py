from .data import Dataset
from .noise_injection import get_noisy_labels
from .metrics import reconstruction_score, log_metrics
from .classifier import Classifier
from label_noise_correction.labelcorrection import LabelCorrectionModel
from datetime import datetime
import mlflow
import pickle
import os
try:
    from tqdm.notebook import tqdm_notebook as tqdm
except ImportError:
    from tqdm import tqdm

class EmpiricalEvaluation:
    """
    Class for running the empirical evaluation.
    """
    def __init__(self, dataset:Dataset, lnc_model:LabelCorrectionModel):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset to use.
        lnc_model : LabelCorrectionModel
            The label correction model to use.
        """
        self.dataset = dataset
        self.lnc_model = lnc_model
        self.exp_name = f'{self.dataset.name}_{self.dataset.sensitive_attr}_{lnc_model.name}'

    def start_experiment(self, noise_injection:bool, classifier:Classifier, noise_type='random', noise_rates=[0.1, 0.2, 0.3, 0.4, 0.5], store_labels=False, store_predictions=False,
                         metrics=['reconstruction_score', 'roc_auc', 'accuracy', 'auc_difference', 'demographic_parity_difference', 'equalized_odds_difference', 'predictive_equality_difference', 'equal_opportunity_difference'],
                         classification_thresholds=[0.2, 0.5, 0.8]):
        """
        Run the empirical evaluation on the considered dataset, using the chosen label noise correction method.
        The results are logged to mlflow.

        Parameters
        ----------
        noise_injection : bool
            Whether to inject noise or not.
        classifier : Classifier
            The classifier to use.
        noise_type : str, optional
            The type of noise to inject. The default is 'random'. Only used if noise_injection is True.
        noise_rates : list, optional
            The noise rates to use. The default is [0.1, 0.2, 0.3, 0.4, 0.5]. Only used if noise_injection is True. 
            Must be a list of floats between 0 and 1.
        metrics : list, optional    
            The metrics to log. 
            The supported metrics are 'reconstruction_score', 'roc_auc', 'accuracy', 'auc_difference', 'demographic_parity_difference', 'equalized_odds_difference', 'predictive_equality_difference' and 'equal_opportunity_difference'.
        classification_thresholds : list, optional
            The classification thresholds to use for calculating the metrics that depend on the confusion matrix.
            The default is [0.2, 0.5, 0.8]. Must be a list of floats between 0 and 1.
        """
        experiment = f'{self.exp_name}_{noise_type}' if noise_injection else self.exp_name
        print(f'Starting experiment: {experiment}')
        mlflow.set_experiment(experiment)  

        run_tag = f'{datetime.now().strftime("%d%m%Y_%H%M%S")}'

        # using standard ML datasets
        if noise_injection:
            print(f'Injecting {noise_type} noise at rates: {noise_rates}')
            for noise_rate in tqdm(noise_rates):
                self.noise_injection_experiment(noise_rate, noise_type, classifier, run_tag, metrics, classification_thresholds, store_labels, store_predictions)
        # using fairness benchmark datasets
        else:
            self.fairness_benchmark_experiment(classifier, run_tag, metrics, classification_thresholds, store_labels, store_predictions)


    def noise_injection_experiment(self, noise_rate, noise_type, classifier:Classifier, run_tag, metrics, classification_thresholds, store_labels, store_predictions):
        # inject noise
        y_train_noisy = get_noisy_labels(noise_type, noise_rate, self.dataset, 'train', store_labels)
        y_test_noisy = get_noisy_labels(noise_type, noise_rate, self.dataset, 'test', store_labels)

        # correct labels
        y_train_corrected = self.lnc_model.correct(self.dataset.get_features('train'), y_train_noisy)
        y_test_corrected = self.lnc_model.correct(self.dataset.get_features('test'), y_test_noisy)

        if store_labels:
            dir = f'data/{self.dataset.name}_{self.dataset.sensitive_attr}/{noise_type}/{noise_rate}'
            y_train_corrected.to_csv(f'{dir}/train_labels_{self.lnc_model.name}_corrected.csv', index=True)
            y_test_corrected.to_csv(f'{dir}/test_labels_{self.lnc_model.name}_corrected.csv', index=True)

        if 'reconstruction_score' in metrics:
            r = reconstruction_score(self.dataset.get_labels(), y_train_corrected, y_test_corrected)

        for test_set in ['original', 'noisy', 'corrected']:
            for train_set in ['original', 'noisy', 'corrected']:
                with mlflow.start_run(tags={'train_set': train_set, 'test_set': test_set, 'run': run_tag}) as run:
                    # log parameters
                    mlflow.log_param('dataset', self.dataset.name)
                    mlflow.log_param('senstive_attr', self.dataset.sensitive_attr)
                    mlflow.log_param('test_size', self.dataset.test_size)
                    mlflow.log_param('noise_rate', noise_rate)
                    mlflow.log_param('noise_type', noise_type)
                    mlflow.log_param('classifier', classifier.name)
                    self.lnc_model.log_params()
                    
                    # log reconstruction score
                    if 'reconstruction_score' in metrics:
                        mlflow.log_metric('reconstruction_score', r)

                    # train classifier and get predictions
                    if train_set == 'original':
                        y_pred_proba = classifier.fit_predict(self.dataset.get_features('train'), self.dataset.get_labels('train'), self.dataset.get_features('test'))
                    elif train_set == 'noisy':
                        y_pred_proba = classifier.fit_predict(self.dataset.get_features('train'), y_train_noisy, self.dataset.get_features('test'))
                    else:
                        if y_train_corrected.unique().shape[0] == 1:
                            print('After noise correction all labels are the same. Skipping classifier training.')
                            continue
                        else:
                            y_pred_proba = classifier.fit_predict(self.dataset.get_features('train'), y_train_corrected, self.dataset.get_features('test'))
                    
                    # log predictions
                    if store_predictions:
                        dir = f'predictions/{self.dataset.name}_{self.dataset.sensitive_attr}/{test_set}/{noise_type}/{noise_rate}'
                        if not os.path.exists(dir):
                            os.makedirs(dir)

                        filename = self.lnc_model.name if train_set == 'corrected' else train_set

                        with open(f'{dir}/{filename}.pkl', 'wb') as f:
                            pickle.dump(y_pred_proba, f)


                    if test_set == 'original':
                        log_metrics(self.dataset.get_labels('test'), y_pred_proba, self.dataset.get_sensitive_attr('test'), metrics, classification_thresholds)
                    elif test_set == 'noisy':
                        log_metrics(y_test_noisy, y_pred_proba, self.dataset.get_sensitive_attr('test'), metrics, classification_thresholds)
                    else:                            
                        log_metrics(y_test_corrected, y_pred_proba, self.dataset.get_sensitive_attr('test'), metrics, classification_thresholds)

    def fairness_benchmark_experiment(self, classifier:Classifier, run_tag, metrics, classification_thresholds, store_labels=False, store_predictions=False):
        y_train_corrected = self.lnc_model.correct(self.dataset.get_features('train'), self.dataset.get_labels('train'))
        y_test_corrected = self.lnc_model.correct(self.dataset.get_features('test'), self.dataset.get_labels('test'))

        if store_labels:
            dir = f'data/{self.dataset.name}_{self.dataset.sensitive_attr}/originally_biased'
            if not os.path.exists(dir):
                os.makedirs(dir)
            y_train_corrected.to_csv(f'{dir}/train_labels_{self.lnc_model.name}_corrected.csv', index=True)
            y_test_corrected.to_csv(f'{dir}/test_labels_{self.lnc_model.name}_corrected.csv', index=True)
        
        for test_set in ['noisy', 'corrected']:
            for train_set in ['noisy', 'corrected']:
                with mlflow.start_run(tags={'train_set': train_set, 'test_set': test_set, 'run': run_tag}) as run:
                    # log parameters
                    mlflow.log_param('dataset', self.dataset.name)
                    mlflow.log_param('senstive_attr', self.dataset.sensitive_attr)
                    mlflow.log_param('test_size', self.dataset.test_size)
                    mlflow.log_param('classifier', classifier.name)
                    self.lnc_model.log_params()

                    # train classifier and get predictions
                    if train_set == 'noisy':
                        y_pred_proba = classifier.fit_predict(self.dataset.get_features('train'), self.dataset.get_labels('train'), self.dataset.get_features('test'))
                    else:
                        if y_train_corrected.unique().shape[0] == 1:
                            print('After noise correction all labels are the same, skipping classifier training.')
                            continue
                        else:
                            y_pred_proba = classifier.fit_predict(self.dataset.get_features('train'), y_train_corrected, self.dataset.get_features('test'))
                    
                    # log predictions
                    if store_predictions:
                        dir = f'predictions/{self.dataset.name}_{self.dataset.sensitive_attr}/{test_set}/originally_biased'
                        if not os.path.exists(dir):
                            os.makedirs(dir)

                        filename = 'noisy' if train_set == 'noisy' else self.lnc_model.name

                        with open(f'{dir}/{filename}.pkl', 'wb') as f:
                            pickle.dump(y_pred_proba, f)

                    if test_set == 'noisy':
                        log_metrics(self.dataset.get_labels('test'), y_pred_proba, self.dataset.get_sensitive_attr('test'), metrics, classification_thresholds)
                    else:
                        log_metrics(y_test_corrected, y_pred_proba, self.dataset.get_sensitive_attr('test'), metrics, classification_thresholds)


