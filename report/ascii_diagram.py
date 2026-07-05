import textwrap

BOX_WIDTH = 46


def _wrap(text, width):
    return textwrap.wrap(text, width=width) or ['']


def _make_box(title, content, width=BOX_WIDTH):
    inner = width - 4
    lines = [f"[{title}]"]
    lines.extend(_wrap(content, inner))

    top = '┌' + '─' * (width - 2) + '┐'
    bottom = '└' + '─' * (width - 2) + '┘'
    body = [f"│ {line.ljust(inner)} │" for line in lines]

    return [top] + body + [bottom]


def _arrow():
    return ['│'.center(BOX_WIDTH), '▼'.center(BOX_WIDTH)]


def draw_network(architecture, spec):
    """
    Builds a simple top-to-bottom ASCII stack of the resolved blueprint.
    architecture: the Architecture fact (has 'family')
    spec: the merged dict already built in format_report()
    """
    family = architecture.get('family', 'unknown').upper().replace('_', ' ')

    sections = [
        ("INPUT",  spec.get('input_layer')),
        ("HIDDEN", spec.get('hidden_layers')),
        ("WIDTH",  spec.get('width')),
        ("OUTPUT", spec.get('output_layer')),
        ("ACTIVATION", spec.get('output_activation')),
    ]

    # drop any section with no data instead of printing an empty box
    sections = [(t, c) for t, c in sections if c]

    lines = []
    lines.append(f"  {family}".center(BOX_WIDTH))
    lines.append('')

    for i, (title, content) in enumerate(sections):
        lines.extend(_make_box(title, content))
        if i != len(sections) - 1:
            lines.extend(_arrow())

    return '\n'.join(lines)