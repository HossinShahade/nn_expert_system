from colorama import Fore, Style, init
init()


def format_report(architecture, blueprints):
    """
    Collects all Blueprint facts, resolves conflicting values,
    and prints the final architecture spec.
    """

    # Resolve each blueprint field to a single deterministic value.
    # Later facts overwrite earlier ones for the same field.
    spec = {}
    notes = []

    for bp in blueprints:
        for key, field in [
            ('input_layer', 'input_layer'),
            ('hidden_layers', 'hidden_layers'),
            ('width', 'width'),
            ('output_layer', 'output_layer'),
            ('output_activation', 'output_activation'),
            ('activation', 'activation'),
            ('normalization', 'normalization'),
            ('dropout_rate', 'dropout_rate'),
            ('dropout_placement', 'dropout_placement'),
            ('loss', 'loss'),
            ('optimizer', 'optimizer'),
            ('lr', 'lr'),
            ('lr_schedule', 'lr_schedule'),
            ('init', 'init'),
        ]:
            val = bp.get(field)
            if val:
                spec[key] = val

        note = bp.get('notes')
        if note and note not in notes:
            notes.append(note)

    # Print
    print(f"\n{Fore.CYAN}{'='*65}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  NEURAL NETWORK ARCHITECTURE RECOMMENDATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}\n")

    if architecture:
        fam = architecture.get('family', 'Unknown').upper().replace('_', ' ')
        print(f"  {Fore.YELLOW}Architecture Family:{Style.RESET_ALL} {fam}\n")

    sections = [
        ("INPUT LAYER",        "input_layer",       Fore.BLUE),
        ("HIDDEN LAYERS",      "hidden_layers",     Fore.BLUE),
        ("WIDTH",              "width",             Fore.BLUE),
        ("OUTPUT LAYER",       "output_layer",      Fore.BLUE),
        ("OUTPUT ACTIVATION",  "output_activation", Fore.GREEN),
        ("HIDDEN ACTIVATION",  "activation",        Fore.GREEN),
        ("NORMALIZATION",      "normalization",     Fore.GREEN),
        ("DROPOUT RATE",       "dropout_rate",      Fore.GREEN),
        ("DROPOUT PLACEMENT",  "dropout_placement", Fore.GREEN),
        ("LOSS FUNCTION",      "loss",              Fore.RED),
        ("OPTIMIZER",          "optimizer",         Fore.YELLOW),
        ("LEARNING RATE",      "lr",                Fore.YELLOW),
        ("LR SCHEDULE",        "lr_schedule",       Fore.YELLOW),
        ("INITIALIZATION",     "init",              Fore.MAGENTA),
    ]

    for label, key, color in sections:
        val = spec.get(key)
        if val:
            print(f"  {color}{label}{Style.RESET_ALL}")
            print(f"  └─ {val}\n")

    if notes:
        print(f"  {Fore.RED}NOTES / WARNINGS{Style.RESET_ALL}")
        for note in notes:
            print(f"  ⚠  {note}")
        print()

    print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}\n")


