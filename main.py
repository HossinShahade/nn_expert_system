from cli.questions import ask_questions
from engine.engine import run_engine
from report.formatter import format_report


def main():
    # Step 1: Ask the user questions
    answers = ask_questions()

    print("\nRunning inference engine...\n")

    # Step 2: Run the expert system
    architecture, blueprints = run_engine(answers)

    # Step 3: Print the report
    format_report(architecture, blueprints)


if __name__ == "__main__":
    main()