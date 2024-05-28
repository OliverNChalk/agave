//! Utility structure used for parsing ranges.
use {
    rand::distributions::{Distribution, Uniform},
    std::str::FromStr,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Range {
    pub min: usize,
    pub max: usize,
}

impl Range {
    /// Generates a random number within the range
    pub fn uniform(&self) -> usize {
        if self.min == self.max {
            return self.min;
        }
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(self.min, self.max);
        dist.sample(&mut rng)
    }

    /// Checks if a given number is within the range
    pub fn contains(&self, value: usize) -> bool {
        value >= self.min && value <= self.max
    }
}

impl FromStr for Range {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trim_chars: &[char] = &['[', ']', ' '];
        let stripped = s.trim_matches(trim_chars);
        let parts: Vec<&str> = stripped.split(',').collect();
        if parts.is_empty() || parts.len() > 2 {
            return Err("Range must be in the format '[min,max]' or 'value'".to_string());
        }

        let min = parts[0]
            .trim()
            .parse()
            .map_err(|_| "Invalid minimum value".to_string())?;
        if parts.len() == 1 {
            return Ok(Range { min, max: min });
        }
        let max = parts[1]
            .trim()
            .parse()
            .map_err(|_| "Invalid maximum value".to_string())?;

        if min > max {
            return Err("Min is greater than max.".to_string());
        }
        Ok(Range { min, max })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_within_range() {
        let range = Range { min: 1, max: 1 };
        assert_eq!(range.uniform(), 1);

        let range = Range { min: 10, max: 10 };
        assert_eq!(range.uniform(), 10);

        let range = Range { min: 1, max: 100 };
        let result = range.uniform();
        assert!(
            (1..=100).contains(&result),
            "Result was not within the range: {result}",
        );
    }

    #[test]
    fn test_contains() {
        let range = Range { min: 1, max: 10 };
        assert!(range.contains(1), "Should contain the lower bound");
        assert!(range.contains(5), "Should contain middle values");
        assert!(range.contains(10), "Should contain the upper bound");
        assert!(
            !range.contains(0),
            "Should not contain values below the range"
        );
        assert!(
            !range.contains(11),
            "Should not contain values above the range"
        );
    }

    #[test]
    fn test_from_str_valid() {
        let valid_single = Range::from_str("10");
        assert_eq!(valid_single, Ok(Range { min: 10, max: 10 }));

        let valid_single = Range::from_str("[10]");
        assert_eq!(valid_single, Ok(Range { min: 10, max: 10 }));

        let valid_range = Range::from_str("[1, 10]");
        assert_eq!(valid_range, Ok(Range { min: 1, max: 10 }));

        let valid_range_spaces = Range::from_str(" [ 1 , 10 ] ");
        assert_eq!(valid_range_spaces, Ok(Range { min: 1, max: 10 }));
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_format = Range::from_str("[1, 10, 15]");
        assert!(invalid_format.is_err(), "Should error on incorrect format");

        let no_numbers = Range::from_str("[]");
        assert!(no_numbers.is_err(), "Should error with no numbers");

        let invalid_min = Range::from_str("[abc, 10]");
        assert!(
            invalid_min.is_err(),
            "Should error on non-numeric min value"
        );

        let invalid_max = Range::from_str("[1, abc]");
        assert!(
            invalid_max.is_err(),
            "Should error on non-numeric max value"
        );

        let invalid_order = Range::from_str("[10, 1]");
        assert!(
            invalid_order.is_err(),
            "Should error when min is greater than max"
        );
    }
}
