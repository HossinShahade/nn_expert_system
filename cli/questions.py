QUESTIONS = [
    {
        "field": "modality",
        "text": "What kind of data are you working with?",
        "options": [
            ("image",      "Images or Video"),
            ("text",       "Text or Documents"),
            ("tabular",    "Numbers in a table (CSV, spreadsheet)"),
            ("audio",      "Audio files"),
            ("timeseries", "Time-ordered measurements (sensor, stock, weather)"),
        ]
    },
    {
        "field": "task",
        "text": "What do you want the model to do?",
        "options": [
            ("classify", "Sort data into categories"),
            ("regress",  "Predict a number"),
            ("generate", "Generate new content (text, images)"),
            ("anomaly",  "Find unusual or abnormal samples"),
            ("detect",   "Find and locate objects within an image"),
        ]
    },
    {
        "field": "output_type",
        "text": "What kind of output does your model produce?",
        "options": [
            ("binary",            "Yes/No or True/False (2 options)"),
            ("multiclass",        "One of several categories (3 or more)"),
            ("continuous_single", "A single number (price, temperature)"),
            ("continuous_multi",  "Multiple numbers at once"),
        ]
    },
    {
        "field": "dataset_size",
        "text": "How much labeled training data do you have?",
        "options": [
            ("tiny",   "Under 1,000 samples"),
            ("small",  "1,000 to 10,000 samples"),
            ("medium", "10,000 to 100,000 samples"),
            ("large",  "Over 100,000 samples"),
        ]
    },
    {
        "field": "sequential",
        "text": "Does the ORDER of your data points matter?",
        "options": [
            (True,  "Yes — each sample depends on what came before"),
            (False, "No — each sample is independent"),
        ]
    },
    {
        "field": "spatial",
        "text": "Is there SPATIAL structure in your data (nearby values relate to each other)?",
        "options": [
            (True,  "Yes — like pixels in an image"),
            (False, "No — features are independent columns"),
        ]
    },
    {
        "field": "is_pretrained",
        "text": "Do you have access to a pretrained model for this type of data?",
        "options": [
            (True,  "Yes — I can download one from HuggingFace or torchvision"),
            (False, "No — I need to train from scratch"),
        ]
    },
    {
        "field": "realtime",
        "text": "Does inference need to happen in real time (low latency)?",
        "options": [
            (True,  "Yes — fast response required (mobile, API, live feed)"),
            (False, "No — batch processing is fine"),
        ]
    },
    {
        "field": "compute",
        "text": "What hardware will you train and deploy on?",
        "options": [
            ("cpu",        "CPU only"),
            ("single_gpu", "Single GPU"),
            ("multi_gpu",  "Multiple GPUs or Cloud"),
        ]
    },
    {
        "field": "class_imbalance",
        "text": "Are your output classes balanced (roughly equal samples per class)?",
        "options": [
            (False, "Yes — balanced"),
            (True,  "No — some classes have far more samples than others"),
        ]
    },
    {
        "field": "interpretability",
        "text": "Do you need to explain WHY the model made its decision?",
        "options": [
            (True,  "Yes — medical, legal, or regulated domain"),
            (False, "No — performance is the priority"),
        ]
    },
    {
        "field": "input_shape",
        "text": "How large is each individual input sample?",
        "options": [
            ("small",  "Image <64px · Text <50 tokens · Tabular <20 features · Time series <50 steps"),
            ("medium", "Image 64-224px · Text 50-512 tokens · Tabular 20-200 features · Time series 50-200 steps"),
            ("large",  "Image >224px · Text >512 tokens · Tabular >200 features · Time series >200 steps"),
        ]
    },
]


def ask_questions():
    """
    Loops through all questions, prints options, reads user input.
    Returns a dict of { field_name: value }.
    """
    from colorama import Fore, Style, init
    init()

    print(f"\n{Fore.CYAN}{'='*60}")
    print("  Neural Network Expert System")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    print("Answer the following questions to get your architecture.\n")

    answers = {}

    for i, q in enumerate(QUESTIONS, 1):
        print(f"{Fore.BLUE}Q{i}: {q['text']}{Style.RESET_ALL}")

        for j, (value, label) in enumerate(q['options'], 1):
            print(f"  {Fore.YELLOW}{j}{Style.RESET_ALL}. {label}")

        while True:
            try:
                choice = int(input(f"\n  {Fore.GREEN}Enter number: {Style.RESET_ALL}"))
                if 1 <= choice <= len(q['options']):
                    answers[q['field']] = q['options'][choice - 1][0]
                    print()
                    break
                else:
                    print(f"  {Fore.RED}Please enter a number between 1 and {len(q['options'])}{Style.RESET_ALL}")
            except ValueError:
                print(f"  {Fore.RED}Please enter a valid number{Style.RESET_ALL}")

    return answers