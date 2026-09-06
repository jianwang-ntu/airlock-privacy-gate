# Demo documents

Two synthetic documents, byte-identical to the two `scripts/demo_run.py` uses.
They are the documents the demo video is recorded on.

`payment_instruction.txt` carries a name, an e-mail address, an IBAN, a phone
number, an employee id and the universally published Luhn-valid test card
`4111 1111 1111 1111`. Nothing here belongs to a real person: the corpus this
project is built on is synthetic by construction
(`gretelai/synthetic_pii_finance_multilingual`) and these two documents were
written by hand in the same shape.

`support_ticket.txt` carries a ticket number and an invoice reference and no
personal identifier, which is why the gate treats the two differently.
