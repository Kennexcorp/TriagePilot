"""CLI entry point: read a ticket, print the routed response."""

from graph.build import build_graph

BANNER = "TriagePilot. Type a ticket and press Enter. Ctrl-D or Ctrl-C to quit."
PROMPT = "\n\n\nTicket: "


def main() -> None:
    graph = build_graph()
    print(BANNER)

    while True:
        try:
            ticket_text = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        result = graph.invoke({"ticket_text": ticket_text})
        print(f"\n{result['response']}")


if __name__ == "__main__":
    main()
