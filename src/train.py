from src import baseline, model_ngram, model_gbm, blend

if __name__ == "__main__":
    for module in (baseline, model_ngram, model_gbm, blend):
        module.main()