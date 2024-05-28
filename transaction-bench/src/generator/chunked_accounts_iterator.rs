//! Encapsulates the logic of traversing over the accounts array
//! and generating chunks of accounts in a circular manner.
//! It is generic to simplify unit tests.
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct ChunkedAccountsIterator<'a, T> {
    data: Arc<Vec<T>>,
    begin: usize,
    chunk_sizes: std::slice::Iter<'a, usize>,
}

impl<'a, T> ChunkedAccountsIterator<'a, T> {
    pub fn new(data: Arc<Vec<T>>, begin: usize, chunk_sizes: &'a [usize]) -> Self {
        ChunkedAccountsIterator {
            data,
            begin,
            chunk_sizes: chunk_sizes.iter(),
        }
    }
}

impl<T: Clone> Iterator for ChunkedAccountsIterator<'_, T> {
    type Item = Vec<T>;

    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Option<Self::Item> {
        let current_chunk_size = self.chunk_sizes.next()?;
        let data_len = self.data.len();
        if data_len == 0 {
            return None;
        }
        let mut result = Vec::with_capacity(*current_chunk_size);
        let mut begin = self.begin;
        let mut end = begin.saturating_add(*current_chunk_size);
        self.begin = end % data_len;
        while end > begin {
            let cur_end = std::cmp::min(end, data_len);
            result.extend_from_slice(&self.data[begin..cur_end]);
            begin = cur_end % data_len;
            end -= cur_end;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let data = Arc::new(vec![]);
        let chunk_sizes = [2, 3];
        let mut iterator: ChunkedAccountsIterator<usize> =
            ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_no_chunk_sizes() {
        let data = Arc::new(vec![1, 2]);
        let chunk_sizes: [usize; 0] = [];
        let mut iterator = ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_normal_operation() {
        let data = Arc::new(vec![1, 2, 3, 4]);
        let chunk_sizes = [1, 2, 1];
        let mut iterator = ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert_eq!(iterator.next().unwrap(), vec![1]);
        assert_eq!(iterator.next().unwrap(), vec![2, 3]);
        assert_eq!(iterator.next().unwrap(), vec![4]);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_wrapping_chunks() {
        let data = Arc::new(vec![1, 2, 3]);
        let chunk_sizes = [2, 3, 1];
        let mut iterator = ChunkedAccountsIterator::new(data, 2, &chunk_sizes);
        assert_eq!(iterator.next().unwrap(), vec![3, 1]);
        assert_eq!(iterator.next().unwrap(), vec![2, 3, 1]);
        assert_eq!(iterator.next().unwrap(), vec![2]);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_long_chunk() {
        let data = Arc::new(vec![1, 2]);
        let chunk_sizes = [4, 5, 2];
        let mut iterator = ChunkedAccountsIterator::new(data, 1, &chunk_sizes);
        assert_eq!(iterator.next().unwrap(), vec![2, 1, 2, 1]);
        assert_eq!(iterator.next().unwrap(), vec![2, 1, 2, 1, 2]);
        assert_eq!(iterator.next().unwrap(), vec![1, 2]);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_single_element_multiple_chunks() {
        let data = Arc::new(vec![1]);
        let chunk_sizes = [1, 1, 2];
        let mut iterator = ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert_eq!(iterator.next().unwrap(), vec![1]);
        assert_eq!(iterator.next().unwrap(), vec![1]);
        assert_eq!(iterator.next().unwrap(), vec![1, 1]);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_exact_match_chunks() {
        let data = Arc::new(vec![1, 2, 3]);
        // Perfectly matches the data length
        let chunk_sizes = [1, 2];
        let mut iterator = ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert_eq!(iterator.next().unwrap(), vec![1]);
        assert_eq!(iterator.next().unwrap(), vec![2, 3]);
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_empty_chunk_sizes() {
        let data = Arc::new(vec![1, 2, 3]);
        let chunk_sizes = [];
        let mut iterator = ChunkedAccountsIterator::new(data, 0, &chunk_sizes);
        assert!(iterator.next().is_none());
    }
}
