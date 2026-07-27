//! progress fixed8へ相入玉専用slotを加えるrouting。

use shogi_format::ShogiBoard;

/// progress8ekが使用するLayerStack数。
pub const PROGRESS8EK_NUM_BUCKETS: usize = 9;
/// 相入玉以外で使用する既存progress bucket数。
pub const PROGRESS8EK_PROGRESS_BUCKETS: usize = 8;
/// 相入玉局面を送る専用slot。
pub const PROGRESS8EK_ENTERING_KING_SLOT: u8 = 8;

/// 双方の玉が、それぞれの視点で五段目以上へ進出しているかを返す。
///
/// `Square::rank()`は上側から0始まりなので、先手玉はrank 0..=4、後手玉は
/// rank 4..=8を五段目以上とする。手番には依存しない。
#[inline]
pub fn is_mutual_entering_king(board: &ShogiBoard) -> bool {
    debug_assert!(board.black_king_sq.is_valid());
    debug_assert!(board.white_king_sq.is_valid());
    board.black_king_sq.rank() <= 4 && board.white_king_sq.rank() >= 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::{Color, types::Square};

    fn board(black_rank: u8, white_rank: u8, side_to_move: Color) -> ShogiBoard {
        ShogiBoard {
            black_king_sq: Square::new(4, black_rank),
            white_king_sq: Square::new(4, white_rank),
            side_to_move,
            ..Default::default()
        }
    }

    #[test]
    fn fifth_rank_is_inclusive_for_both_kings() {
        assert!(is_mutual_entering_king(&board(4, 4, Color::Black)));
        assert!(is_mutual_entering_king(&board(3, 5, Color::Black)));
    }

    #[test]
    fn either_king_outside_fifth_rank_rejects_position() {
        assert!(!is_mutual_entering_king(&board(5, 4, Color::Black)));
        assert!(!is_mutual_entering_king(&board(4, 3, Color::Black)));
    }

    #[test]
    fn predicate_does_not_depend_on_side_to_move() {
        let black = board(4, 4, Color::Black);
        let white = board(4, 4, Color::White);
        assert_eq!(
            is_mutual_entering_king(&black),
            is_mutual_entering_king(&white)
        );
    }
}
