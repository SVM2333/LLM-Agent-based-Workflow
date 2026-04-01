"""Example inputs for power dispatch scenarios"""

EXAMPLE_INPUTS = {
    "example1": {
        "title": "Basic Power Dispatch Optimization",
        "description": "5 thermal power units basic dispatch problem",
        "input": "Optimize a power dispatch system with 5 thermal generators, goal is to minimize generation cost while satisfying load demand and unit output constraints."
    },

    "example2": {
        "title": "24-Hour Dispatch Plan",
        "description": "Time-varying dispatch with ramping constraints",
        "input": "Design a 24-hour power dispatch plan with 3 thermal generators, total load demand fluctuates between 500-800MW, goal is to minimize fuel cost while satisfying unit ramping rate limits."
    },

    "example3": {
        "title": "Emission-Constrained Dispatch",
        "description": "Dispatch with emission limits",
        "input": "Optimize dispatch for 10 units, minimize cost while satisfying emission limits, time horizon 24 hours, each unit output range 50-200MW."
    },

    "example4": {
        "title": "Multi-Objective Optimization",
        "description": "Consider both cost and emissions",
        "input": "Optimize dispatch for 8 generators, goal is to minimize generation cost and carbon emissions simultaneously, satisfy 24-hour load demand, considering unit start-up/shut-down costs."
    },

    "example5": {
        "title": "Complex Constraints Dispatch",
        "description": "Multiple constraint types",
        "input": "Optimize 15 mixed units (thermal + hydro) 24-hour dispatch, minimize total cost, constraints include: power balance, unit capacity, ramping rate, minimum on/off time, reservoir capacity limits."
    }
}


def get_example(example_id: str) -> str:
    """Get example input by ID"""
    example = EXAMPLE_INPUTS.get(example_id)
    if example:
        return example["input"]
    return ""


def list_examples():
    """List all examples"""
    for key, value in EXAMPLE_INPUTS.items():
        print(f"\n{key}: {value['title']}")
        print(f"  Description: {value['description']}")
        print(f"  Input: {value['input'][:50]}...")


if __name__ == "__main__":
    list_examples()
