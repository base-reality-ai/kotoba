//! Tiny host-domain module for the embedding example.
//!
//! In a real spawned project this file would be the app's own domain logic.
//! The skeleton ingests this file into `.dm/wiki/` to show dm tracking host
//! code rather than only its kernel internals.

#[derive(Debug, Clone, Copy)]
pub struct Transaction {
    pub amount_cents: i64,
}

pub fn sample_transactions() -> Vec<Transaction> {
    vec![
        Transaction {
            amount_cents: 12_500,
        },
        Transaction {
            amount_cents: -4_000,
        },
        Transaction {
            amount_cents: 1_550,
        },
    ]
}

pub fn projected_balance_cents(transactions: &[Transaction]) -> i64 {
    transactions.iter().map(|tx| tx.amount_cents).sum()
}
