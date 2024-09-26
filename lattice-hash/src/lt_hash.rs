use {
    base64::{display::Base64Display, prelude::BASE64_STANDARD},
    std::fmt,
};

/// A 16-bit, 1024 element lattice-based incremental hash based on blake3
//
// Developer notes:
// - Do not derive Copy because this type is large and copying will not be fast/free.
// - Do not derive Default because hashes do not have a meaningful "default".
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LtHash(pub [u16; LtHash::NUM_ELEMENTS]);

impl LtHash {
    pub const NUM_ELEMENTS: usize = 1024;

    /// Creates a new LtHash from `hasher`
    ///
    /// The caller should hash in all inputs of interest prior to calling.
    #[must_use]
    pub fn with(hasher: &blake3::Hasher) -> Self {
        let mut reader = hasher.finalize_xof();
        let mut inner = [0; Self::NUM_ELEMENTS];
        reader.fill(bytemuck::must_cast_slice_mut(inner.as_mut_slice()));
        Self(inner)
    }

    /// Mixes `other` into `self`
    ///
    /// This can be thought of as akin to 'insert'
    pub fn mix_in(&mut self, other: &Self) {
        for i in 0..self.0.len() {
            self.0[i] = self.0[i].wrapping_add(other.0[i]);
        }
    }

    /// Mixes `other` out of `self`
    ///
    /// This can be thought of as akin to 'remove'
    pub fn mix_out(&mut self, other: &Self) {
        for i in 0..self.0.len() {
            self.0[i] = self.0[i].wrapping_sub(other.0[i]);
        }
    }

    /// Computes a checksum of the LtHash
    pub fn checksum(&self) -> Checksum {
        let hash = blake3::hash(bytemuck::must_cast_slice(&self.0));
        Checksum(hash.into())
    }
}

impl fmt::Display for LtHash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let base64 = Base64Display::new(bytemuck::must_cast_slice(&self.0), &BASE64_STANDARD);
        write!(f, "{base64}")
    }
}

/// A smaller "checksum" of the LtHash, useful when 2 KiB is too large
//
// Developer notes:
// - Do not derive Copy because copying may not be fast/free.
// - Do not derive Default because there is not a meaningful "default".
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Checksum(pub [u8; Checksum::NUM_ELEMENTS]);

impl Checksum {
    pub const NUM_ELEMENTS: usize = 32;
}

impl fmt::Display for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let base64 = Base64Display::new(&self.0, &BASE64_STANDARD);
        write!(f, "{base64}")
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        rand::prelude::*,
        std::ops::{Add, Sub},
    };

    impl LtHash {
        const fn new_zeroed() -> Self {
            Self([0; Self::NUM_ELEMENTS])
        }

        fn new_random() -> Self {
            let mut new = Self::new_zeroed();
            thread_rng().fill(&mut new.0);
            new
        }
    }

    impl Add for LtHash {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self {
            self.mix_in(&rhs);
            self
        }
    }

    impl Sub for LtHash {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self {
            self.mix_out(&rhs);
            self
        }
    }

    impl Copy for LtHash {}

    // Ensure that if you mix-in then mix-out a hash, you get the original value
    #[test]
    fn test_inverse() {
        let a = LtHash::new_random();
        let b = LtHash::new_random();
        assert_eq!(a, a + b - b);
    }

    // Ensure that mixing is commutative
    #[test]
    fn test_commutative() {
        let a = LtHash::new_random();
        let b = LtHash::new_random();
        assert_eq!(a + b, b + a);
    }

    // Ensure that mixing is associative
    #[test]
    fn test_associative() {
        let a = LtHash::new_random();
        let b = LtHash::new_random();
        let c = LtHash::new_random();
        assert_eq!((a + b) + c, a + (b + c));
    }

    // Ensure the correct lattice hash and checksum values are produced
    #[test]
    fn test_hello_world() {
        let expected_hello_lt_hash = LtHash([
            0x8fea, 0x3d16, 0x86b3, 0x9282, 0x445e, 0xc591, 0x8de5, 0xb34b, 0x6e50, 0xc1f8, 0xb74e,
            0x868a, 0x08e9, 0x62c5, 0x674a, 0x0f20, 0x92e9, 0x5f40, 0x780d, 0x595b, 0x2e9a, 0x8733,
            0xd3f6, 0x014d, 0xccfa, 0xb2fe, 0xb62f, 0xef97, 0xd53f, 0x4135, 0x1a24, 0x8c33, 0x88c6,
            0x5676, 0xb58a, 0xe5c6, 0xab24, 0xfebc, 0x1e88, 0x4e5b, 0xc91a, 0x6f33, 0x933f, 0x412d,
            0x4822, 0x82c9, 0x3695, 0x9f69, 0xa107, 0xceb1, 0xff35, 0xe0df, 0x5dbe, 0xc000, 0xa883,
            0xd2df, 0x9a9c, 0x0343, 0x37d1, 0xd74c, 0x6a0e, 0xecbc, 0x6b6e, 0x6c79, 0xac92, 0x0905,
            0xc1cf, 0xaa9d, 0x6969, 0x736e, 0xcf4c, 0x0029, 0xcf70, 0x8f05, 0xde0f, 0x3fc9, 0x1db6,
            0x6d09, 0x2e08, 0xf4aa, 0x7208, 0x2cc1, 0x8cfb, 0x276e, 0xd62e, 0x2211, 0xf254, 0x8518,
            0x4d07, 0x1594, 0xf13f, 0xab12, 0xcc65, 0x4d4a, 0xceba, 0xfe93, 0x589f, 0x9f4e, 0xe7ea,
            0x63a8, 0xe612, 0x4ced, 0x58a5, 0x43b3, 0x39f6, 0x457c, 0x474f, 0x9aff, 0x5124, 0x63f6,
            0x450d, 0x3fc2, 0x9ccf, 0xf0c6, 0xc69f, 0x2bd3, 0x7a5d, 0x9574, 0x2f2c, 0xf934, 0xcc03,
            0x9342, 0x9998, 0x0da9, 0x6dd1, 0x460d, 0x3e00, 0xcdde, 0xf14d, 0x06ec, 0x6b74, 0x9551,
            0x68c4, 0x0f94, 0x4ac6, 0xed49, 0xd886, 0x24cb, 0x2a29, 0xf4a4, 0x3a83, 0x1f81, 0xe97a,
            0xfa1e, 0xb1c5, 0xfcd5, 0xb24c, 0xdb92, 0x2b62, 0xa4f1, 0x498e, 0xf00d, 0x63be, 0x7f6e,
            0x2c33, 0xdc3e, 0xb0fb, 0xe854, 0x8ee3, 0x5d95, 0xc613, 0x670b, 0xf4aa, 0x5570, 0x04bc,
            0xf606, 0x664f, 0xe5ec, 0xd65b, 0x0ea1, 0xf37c, 0x7745, 0x809b, 0x031e, 0xed80, 0x7254,
            0x211b, 0x0cce, 0x94e1, 0x6bf6, 0x95b1, 0x49ba, 0x64c0, 0x8ec9, 0x3b27, 0x5f21, 0xafc8,
            0x3b86, 0x2ea5, 0x8c30, 0x168e, 0xc147, 0x1fd5, 0x1637, 0x88f5, 0x9321, 0x63aa, 0xaae5,
            0x33bb, 0xd983, 0xb09a, 0xf24e, 0xa1e5, 0x2b39, 0xd434, 0x7135, 0x61ed, 0x57ad, 0x5940,
            0xe53f, 0x727d, 0x4882, 0x8c44, 0xa61b, 0x1b9f, 0xcee4, 0xf462, 0xc875, 0xc019, 0x9310,
            0x7dc2, 0xf55c, 0xcb36, 0x9505, 0xebb5, 0x8a2b, 0x2b07, 0x0a36, 0x3890, 0x54c8, 0x5a76,
            0xece7, 0x96f1, 0xe3f7, 0x6d99, 0x83e4, 0xff35, 0x1d04, 0x8783, 0xbf2e, 0xb846, 0x79a9,
            0x69ba, 0xb980, 0x28f6, 0x2325, 0x7d13, 0xc44c, 0xacba, 0x134e, 0xa877, 0x6b67, 0x8027,
            0xba94, 0xf564, 0x2174, 0xf985, 0x91c8, 0xd568, 0x319f, 0x6d4e, 0xa59b, 0xd344, 0x4a67,
            0x801d, 0x7aeb, 0x20c0, 0xba23, 0x9744, 0xdd93, 0x4cc5, 0x1148, 0xdf86, 0xad19, 0x06b7,
            0xa824, 0x8e56, 0x2cab, 0x9ad1, 0x5ec0, 0xd57c, 0x0f2b, 0x8d85, 0x65e2, 0xd9c0, 0xc824,
            0x3cae, 0xed26, 0x5c7c, 0x41f9, 0x4767, 0xf730, 0xe210, 0x2926, 0xb68f, 0xcf36, 0x22b9,
            0x5f1b, 0x4ae4, 0xcdcd, 0xe69a, 0x9f4c, 0x1036, 0x8e7c, 0x48de, 0xee0f, 0xbcbd, 0x6bc7,
            0x067a, 0x35e6, 0x98fa, 0x2dcb, 0xa442, 0xbcd0, 0xa02c, 0xc746, 0x60b9, 0x479e, 0x6f56,
            0xff1a, 0xe6f0, 0xef75, 0x5dad, 0x2096, 0xbd07, 0x96e2, 0x2bc6, 0xee33, 0xd122, 0x05f7,
            0x2177, 0x2dbc, 0x729b, 0xfdf0, 0x2c18, 0x800c, 0xdb7d, 0xfb19, 0x0002, 0x3895, 0x5b72,
            0xfbe7, 0x16ce, 0x671f, 0x2175, 0x7c84, 0xc8dc, 0x9690, 0xf594, 0x31b4, 0x47f3, 0xe3f2,
            0x8911, 0x747d, 0x25c2, 0x480a, 0x16ff, 0xba50, 0x8bcb, 0xe9d7, 0xec54, 0x7df4, 0x4b9a,
            0xf4bb, 0x3100, 0x86cc, 0x62c2, 0x9b73, 0x06d7, 0x157b, 0x0922, 0xab9e, 0x83a6, 0x2f28,
            0x30ce, 0x3eff, 0x5134, 0xc9d5, 0x74ae, 0x295c, 0x9af8, 0x482a, 0x61dc, 0xe555, 0x9c7c,
            0x57de, 0xfe56, 0xd898, 0x19c6, 0x444f, 0x9636, 0x9297, 0xea84, 0xeaba, 0xce24, 0x6dc0,
            0x14c3, 0x6e7d, 0x2a65, 0x3bb5, 0x679d, 0x22a1, 0x8ea1, 0xc564, 0xca61, 0x0b2a, 0x38ea,
            0xe029, 0xcf07, 0x4280, 0xff2a, 0x8697, 0x8d30, 0x185b, 0x919a, 0x8f7c, 0x046c, 0x9390,
            0x50ab, 0xcb51, 0x2334, 0x616f, 0x998f, 0x1d2d, 0xd294, 0x74f1, 0x822c, 0xe50d, 0xdcc6,
            0xbafc, 0x7d92, 0xe202, 0xe28e, 0x2e19, 0xecaa, 0x7cf5, 0x25aa, 0x7a1a, 0x389a, 0xc189,
            0x6af0, 0x6fa3, 0x16c3, 0xa318, 0x8cb5, 0x348e, 0x627b, 0xd144, 0x7d8d, 0xc43c, 0xca5b,
            0xf4bd, 0xb174, 0x4734, 0x3520, 0xbeb9, 0x4f79, 0xa628, 0xe4bd, 0x1bc7, 0xa9f4, 0x3ad2,
            0x959b, 0xe178, 0x1ba2, 0x48bb, 0x5e79, 0xd594, 0xf41e, 0x78ce, 0x685c, 0x79d4, 0xedae,
            0xe11d, 0x2172, 0xb9ab, 0x5ca2, 0xf9ff, 0x2812, 0x66b7, 0xed6d, 0x7eff, 0x960f, 0x4844,
            0x9484, 0x504a, 0x5b29, 0xca8b, 0xdafd, 0xa6b7, 0xef3a, 0xe2e0, 0xa137, 0x1b05, 0x16c2,
            0xefbd, 0x06ac, 0xf3f1, 0xa94f, 0xcade, 0x7087, 0x2ec9, 0x6543, 0x49a1, 0xf4c3, 0x3157,
            0xed65, 0xfc85, 0xefd4, 0x30b8, 0xa5e8, 0x093f, 0xcbe2, 0x8e2b, 0x2fd4, 0xae39, 0x3e37,
            0x37c5, 0xf02f, 0xf643, 0xc03e, 0xe4d0, 0xe305, 0xfd1a, 0x698d, 0x1285, 0x19de, 0x1582,
            0x251f, 0xe136, 0x3eec, 0x862b, 0xbf4d, 0xab67, 0x0c90, 0x3eb5, 0x58d0, 0xc300, 0x7f93,
            0x03e1, 0xf2f9, 0x78fd, 0x93b6, 0x5add, 0x865a, 0x8b20, 0x89e4, 0x7585, 0x6e40, 0x5a8a,
            0x8623, 0x7335, 0xa9e1, 0xfecf, 0x83cb, 0xe9de, 0xf07c, 0x36ca, 0x5a7b, 0x9fff, 0xe419,
            0x8e48, 0xa704, 0xbcab, 0x44ae, 0x6dfa, 0x810c, 0x94f4, 0x62fb, 0xa34e, 0xa9a5, 0x1d13,
            0x98a9, 0x88ba, 0x7bc2, 0x7a59, 0x188a, 0x1855, 0xd27d, 0x6781, 0xcf08, 0xde49, 0x5588,
            0x5c8b, 0x1f4a, 0xd22b, 0x3959, 0xe754, 0xf071, 0xdfc2, 0xf352, 0x255c, 0x2d36, 0x59d0,
            0x4621, 0x1ed0, 0xa0b5, 0x457d, 0xd3d7, 0xd137, 0x10ca, 0xeeb1, 0xec30, 0x96af, 0x9be5,
            0x2181, 0xe570, 0x8a33, 0x137e, 0x861e, 0xd155, 0x950d, 0xc6e4, 0x5c1f, 0xe4dc, 0x4466,
            0x7078, 0x75a5, 0x7a51, 0x1339, 0xa1a8, 0xcb89, 0xf383, 0xabf0, 0x0170, 0xbb1d, 0xea76,
            0xe491, 0xf911, 0xdc42, 0xec04, 0x82b8, 0xeadd, 0xc890, 0x505c, 0xafa7, 0x42cb, 0xfd99,
            0x127e, 0x0724, 0xd4f9, 0x94ef, 0xf060, 0x67fe, 0x038d, 0x2876, 0xb812, 0xbf05, 0xe904,
            0x003e, 0x2ee4, 0xe8f5, 0x0a66, 0xd790, 0x3ccc, 0x28be, 0xdbc2, 0x073c, 0xd4a5, 0x904c,
            0x60ad, 0x4f67, 0x77ac, 0xae49, 0x2d6c, 0x9220, 0xde9c, 0x2a2b, 0xf99c, 0xb54f, 0x8290,
            0x2e7d, 0x0ca1, 0xf79b, 0xc6ff, 0x3e6e, 0x8eb4, 0x66b1, 0xc6e6, 0x600f, 0xda08, 0xa933,
            0x2cad, 0x308a, 0x93f2, 0x4f70, 0x72d3, 0x56e0, 0x4ddd, 0x682c, 0x589f, 0xd461, 0x06ad,
            0x4e9a, 0x1af7, 0x901c, 0xa1d4, 0xb990, 0xbbcc, 0xdcbb, 0xe46f, 0xe585, 0x9800, 0x86e6,
            0xa735, 0xac0f, 0xb666, 0xaeac, 0x6e00, 0x8b36, 0xc4ce, 0x7261, 0xf078, 0xb42a, 0x86fb,
            0xd4d8, 0x1402, 0xd7ac, 0x69c6, 0x8b29, 0x66ce, 0x512d, 0x93f8, 0x811b, 0x7b2c, 0x1a3b,
            0x88fb, 0x8ca2, 0x197e, 0xbd7b, 0x5c5c, 0xf2c3, 0x803b, 0xe9f2, 0x6fd2, 0x8c05, 0x6966,
            0x2249, 0xceab, 0xe42b, 0x8195, 0x9ddc, 0x79ee, 0x1e35, 0x3fd4, 0x6fc4, 0x9b26, 0x85b0,
            0x45a4, 0x5a6b, 0xf43b, 0x0f07, 0x3104, 0x463d, 0x710a, 0x288e, 0x0dcd, 0x8f1a, 0xa307,
            0x6790, 0x1f2e, 0x991a, 0x7fcc, 0x241a, 0x80d9, 0x9f22, 0xac19, 0x0015, 0x5690, 0x45ba,
            0x4a3f, 0x84f1, 0x01c5, 0xc2b8, 0xa512, 0xffc0, 0xebbd, 0x3c5f, 0x66dc, 0x9fdd, 0xe066,
            0x5b39, 0x2fa1, 0x9432, 0xad65, 0xf397, 0x528a, 0x0c94, 0xe646, 0xbeb5, 0xe91c, 0x7d24,
            0x305c, 0x2c7b, 0x3f93, 0x860e, 0x6e39, 0x953a, 0xb010, 0xbb1b, 0x15a2, 0x369b, 0xf840,
            0xa258, 0xb39a, 0x522b, 0xedbb, 0x7fb9, 0xb94c, 0x45d0, 0x34c0, 0xd516, 0xb52d, 0xdce1,
            0x35e4, 0x3801, 0x3e5c, 0x6826, 0x3b4e, 0xc688, 0xe612, 0x64a8, 0x7898, 0xd07f, 0xa93e,
            0x0f42, 0x9392, 0xa877, 0xd68f, 0xd947, 0x7615, 0xac5e, 0x6f1c, 0x3a42, 0x04c8, 0x993e,
            0x53e5, 0x272e, 0x3021, 0xa3d2, 0xfc24, 0xbd1e, 0xf109, 0x3b8f, 0x6566, 0x48f9, 0x4ef5,
            0x777d, 0xcbaa, 0x029e, 0x8867, 0xda07, 0xa941, 0xeb45, 0x8ad2, 0x9c78, 0xa7c9, 0xdf67,
            0x2ec0, 0x8c0b, 0x6827, 0x18ca, 0x78c2, 0xc9df, 0x8a0e, 0x2aae, 0x4e31, 0xa7ec, 0xd0e5,
            0x748c, 0x1556, 0x44ad, 0xec45, 0x9e48, 0x13d1, 0x74ae, 0x1382, 0x6fdd, 0x6d15, 0x39b9,
            0x4a8a, 0xe31d, 0x4732, 0xb215, 0x5b5e, 0x5b7a, 0x5981, 0x4e94, 0x2ccd, 0x12b6, 0x5072,
            0x4e2b, 0x078f, 0x6896, 0xec47, 0x1165, 0x2625, 0x7fd3, 0xe652, 0xb05f, 0x6fc8, 0xfcb0,
            0xf199, 0xef36, 0x89db, 0xb274, 0x3e7c, 0x9985, 0xbc7a, 0xbd5e, 0x9f19, 0x6068, 0x47f2,
            0xc8db, 0x8025, 0x3e28, 0xf0b2, 0xbad1, 0x1237, 0x3b1d, 0xe2fc, 0x24b7, 0xb8b8, 0x4d82,
            0x5adc, 0x16b4, 0x1bb7, 0xedec, 0x9f94, 0x3557, 0x4ce4, 0x9995, 0xec62, 0xce8e, 0x597e,
            0x0161, 0x12f7, 0xa4d3, 0x98c7, 0xaede, 0x7e2d, 0xaa32, 0x98e4, 0xbfd7, 0x7e5a, 0x9507,
            0x8900, 0x1f5a, 0x46f5, 0x64cf, 0x6885, 0x6977, 0x26c4, 0xd94a, 0xe454, 0xcd75, 0xeda1,
            0x476b, 0x697c, 0xe522, 0x4ab9, 0x9e88, 0xde52, 0x67e4, 0xb170, 0x3270, 0x6291, 0x2422,
            0x95bb, 0xcf27, 0x90da, 0x12b2, 0x1305, 0x029b, 0x8427, 0x52e5, 0x3e64, 0x7a88, 0xd34d,
            0x68ee, 0x6099, 0xae6d, 0x622f, 0x1237, 0x33bd, 0x0143, 0x1e1c, 0xd463, 0xda74, 0x7272,
            0xa794, 0x1714, 0x8ec6, 0xf919, 0xdb4c, 0x60d7, 0xa3ae, 0xe336, 0x12bf, 0xc469, 0xfc67,
            0x9037, 0xcb6a, 0x5ebd, 0x85b5, 0x6c11, 0xa54e, 0x7e7f, 0xec0d, 0x46e5, 0x43ec, 0x6bf5,
            0x086f, 0x9421, 0xf5f7, 0xdbdf, 0x9994, 0x072c, 0xe5d9, 0x19a5, 0x8458, 0xec68, 0xba3f,
            0x9924,
        ]);
        let expected_hello_checksum = Checksum([
            79, 156, 26, 184, 156, 205, 94, 208, 182, 235, 33, 147, 111, 153, 229, 152, 207, 133,
            75, 109, 182, 198, 119, 61, 11, 81, 41, 70, 24, 87, 100, 85,
        ]);

        let expected_world_lt_hash = LtHash([
            0x56dc, 0x1d98, 0x5420, 0x810d, 0x936f, 0x1011, 0xa2ff, 0x6681, 0x637e, 0x9f2c, 0x0024,
            0xebd4, 0xe5f2, 0x3382, 0xd48b, 0x209e, 0xb031, 0xe7a5, 0x026f, 0x55f1, 0xc0cf, 0xe566,
            0x9eb0, 0x0a41, 0x3eb1, 0x3d36, 0x1b7c, 0x83ca, 0x9aa6, 0x2264, 0x8794, 0xfb85, 0x71e0,
            0x64c9, 0x227c, 0xed27, 0x09e0, 0xe5d5, 0xc8da, 0x88a5, 0x8b49, 0xf5a5, 0x3137, 0xbeed,
            0xca0e, 0x7690, 0x0570, 0xa5de, 0x4e0b, 0x4827, 0x4ae4, 0x2dad, 0x0ce4, 0xd56f, 0x9819,
            0x5d4e, 0xe93a, 0x0024, 0xb7b2, 0xc7ba, 0xa00c, 0x6709, 0x1d26, 0x53d3, 0x17b1, 0xebdf,
            0xb18f, 0xb30a, 0x3d6b, 0x1d75, 0x26a0, 0x260e, 0x6585, 0x2ba6, 0xc88d, 0x70ef, 0xf6f4,
            0x8b7f, 0xc03b, 0x285b, 0x997b, 0x933e, 0xf139, 0xe097, 0x3eff, 0xd9f7, 0x605a, 0xaeec,
            0xee8d, 0x1527, 0x3bff, 0x7081, 0xda28, 0x4c0f, 0x44b0, 0xb7d0, 0x8f9b, 0xa657, 0x8e47,
            0xa405, 0x5507, 0xe5f9, 0x52ed, 0xc4e1, 0x300c, 0x0db3, 0xbf93, 0xfddd, 0x8f21, 0x10c5,
            0x4bfd, 0x5f13, 0xe136, 0xd72f, 0x1822, 0xb424, 0x996f, 0x8fdd, 0x0703, 0xa57f, 0x7923,
            0x0755, 0x7aee, 0x168d, 0x1525, 0xf912, 0xb48d, 0xfb9e, 0xd606, 0xb2ce, 0x98ef, 0x20fb,
            0xd21a, 0x8261, 0xd6db, 0x61bf, 0xdbc6, 0x02b1, 0x45e9, 0x1ffa, 0x071f, 0xa2c0, 0x74a8,
            0xae54, 0x59e1, 0xe2dc, 0x0ec9, 0x35ac, 0xbbb0, 0x5938, 0x2210, 0xcf9e, 0x2d9f, 0x7e01,
            0x2ab7, 0xd7d8, 0x8e36, 0x6b09, 0x262c, 0xb017, 0x9b6e, 0x1455, 0x7401, 0x8a8a, 0x6491,
            0x9de9, 0x7856, 0x8fb3, 0x8fcb, 0x3c05, 0x3e74, 0x40a4, 0x682a, 0x1a67, 0x9888, 0xb949,
            0xbb75, 0x6ef9, 0xc457, 0xa83a, 0x7965, 0x159e, 0xa415, 0x1c6b, 0x1b94, 0xaa10, 0x137d,
            0xbc3a, 0xc6bd, 0xf303, 0x7758, 0xc8da, 0xf5a3, 0x5826, 0x2b48, 0x9852, 0x3033, 0xfa85,
            0x3f85, 0x9b38, 0xd409, 0x4813, 0x36b2, 0x43d7, 0xdc0a, 0xfb54, 0x22b2, 0xf1e1, 0xfe5a,
            0x44ff, 0x217c, 0x158d, 0x2041, 0x7d2a, 0x4a78, 0xfc39, 0xb7db, 0x4786, 0xf8ee, 0xc353,
            0x96c2, 0x7be2, 0xd18d, 0x0407, 0x7b0e, 0x04f5, 0x3c63, 0x415e, 0xb1d1, 0x31cc, 0x25ac,
            0x9d8a, 0x4845, 0xd2b4, 0x0cdd, 0xf9a4, 0xae8f, 0x7fe5, 0x2285, 0xa749, 0x43cb, 0x16ae,
            0x09a9, 0xbd32, 0x923c, 0x2825, 0xbe21, 0xfa66, 0x2638, 0x3435, 0x6d79, 0xdf4b, 0xaab4,
            0xf2b1, 0x08f4, 0x64fd, 0x7364, 0x14e4, 0x1457, 0xbce3, 0xe114, 0xeccb, 0x2490, 0xae79,
            0x7448, 0x6310, 0xeff6, 0x2bb1, 0x79e7, 0xf5ae, 0xab40, 0xff6d, 0x889b, 0xe5f5, 0x69ee,
            0x3298, 0x512a, 0x2573, 0xf85c, 0xc69a, 0xb142, 0x3ed0, 0x7b9d, 0xc7a5, 0xea5d, 0xd085,
            0x4e99, 0xaf95, 0x404b, 0x8aca, 0x870f, 0x098a, 0x7c9c, 0x30cf, 0x3e16, 0x9010, 0xa94b,
            0x3cca, 0x00bc, 0xddb8, 0xbf1b, 0xc61a, 0x7121, 0xd668, 0xf4ba, 0xb339, 0xa66c, 0xd5b9,
            0x557c, 0x70a0, 0x34e4, 0x43a5, 0x9c32, 0x2e94, 0xa47f, 0x0b21, 0xb594, 0xb483, 0xf823,
            0x8c56, 0x9ee9, 0x71aa, 0xf97c, 0x1c62, 0xe003, 0xcbbe, 0xca8f, 0x58e5, 0xcbee, 0x758e,
            0x5511, 0x38da, 0x7816, 0xd6a1, 0x4550, 0x09e9, 0x682f, 0xf2ca, 0x5ea1, 0x58c2, 0x78ed,
            0xb630, 0xee80, 0xa2df, 0xa890, 0x8b42, 0x83d0, 0x7ec6, 0xa87e, 0x896c, 0xf649, 0x173d,
            0x4950, 0x5d0a, 0xd1a8, 0x7376, 0x4a4a, 0xe53f, 0x447d, 0x6efd, 0xd202, 0x1da3, 0x4825,
            0xd44b, 0x4343, 0xa1a9, 0x8aac, 0x5b50, 0xc8e6, 0x8086, 0xd64f, 0xd077, 0x76f0, 0x9443,
            0xcd70, 0x950d, 0x0369, 0xf1be, 0xb771, 0x5222, 0x4b40, 0x4846, 0x3fab, 0x1d5d, 0xc69d,
            0xa200, 0xe217, 0xb8bd, 0x2ef7, 0xed6b, 0xa78c, 0xe978, 0x0e16, 0x72bf, 0x05a3, 0xdcb4,
            0x4024, 0xfca2, 0x0219, 0x0d3e, 0xa83f, 0x6127, 0x33ab, 0x3ae5, 0xe7a1, 0x2e76, 0xf6f5,
            0xbee1, 0xa712, 0xab89, 0xf058, 0x71ed, 0xd39e, 0xa383, 0x5f64, 0xe2b6, 0xbe86, 0xee47,
            0x5bd8, 0x1536, 0xc6ed, 0x1c40, 0x836d, 0xcc40, 0x18ff, 0xe30a, 0xae2c, 0xc709, 0x7b40,
            0xddf8, 0x7b72, 0x97da, 0x3f71, 0x6dba, 0x578b, 0x980a, 0x2e0e, 0xd0c0, 0x871f, 0xde9b,
            0xa821, 0x1a41, 0xbff0, 0x04cb, 0x40d6, 0x9942, 0xf717, 0x2c1a, 0x65f9, 0xae3d, 0x9e4e,
            0x3ca6, 0x2d53, 0x3f6e, 0xc886, 0x5bbc, 0x9936, 0x09de, 0xb4ab, 0xc044, 0xa7a0, 0x8c37,
            0x383a, 0x3ab9, 0xcd16, 0x33c2, 0x908e, 0x75c3, 0x51da, 0xcb86, 0x4640, 0xe2b7, 0xbc2f,
            0x1bbb, 0xc1c0, 0xc4ce, 0x821d, 0x0a46, 0x178c, 0x1291, 0xfe6e, 0xd15f, 0x8d3e, 0x9d01,
            0x79b2, 0xfe4c, 0x75eb, 0x176c, 0x6be7, 0x6efa, 0xdcc6, 0x2127, 0xef2b, 0xb83a, 0xe10b,
            0x3206, 0xc2fe, 0x1a3d, 0x62c8, 0xf55e, 0xc594, 0x81ba, 0x0188, 0x962a, 0x0f1c, 0x2489,
            0xb3ca, 0x0d9a, 0xca06, 0xfe37, 0x2cb0, 0x87a1, 0xd33b, 0x31b0, 0x1efe, 0x08f2, 0xc55a,
            0xcb8a, 0x1633, 0x9df2, 0xc468, 0xd5e3, 0x3117, 0x3333, 0x488f, 0x4a9d, 0xc68f, 0x73f9,
            0xa82d, 0xe1af, 0xeb4e, 0xe41b, 0x33f5, 0x051f, 0x7592, 0x0528, 0x7aee, 0xc3eb, 0x7010,
            0x03f4, 0xaba4, 0x3e8f, 0x4abd, 0x2b41, 0x5390, 0x21a1, 0x6dc6, 0xd828, 0xa9b4, 0xc63a,
            0x3ab3, 0x14aa, 0xdc3a, 0x513f, 0x9886, 0x0000, 0x1169, 0xbba0, 0xb2fe, 0x4b09, 0x0198,
            0xcfff, 0xb898, 0x8cfe, 0x3def, 0x0b4b, 0xc154, 0x2491, 0x28d7, 0x757f, 0x06c5, 0x98c5,
            0x2dfa, 0xc068, 0xc74d, 0x521e, 0x70d5, 0xde35, 0x7718, 0xddf8, 0xa387, 0x807d, 0x0056,
            0x697b, 0x3043, 0x4ec8, 0xc2be, 0xa867, 0x0555, 0x2d3f, 0xc9f1, 0xfe7c, 0xe851, 0x5b85,
            0x2175, 0x741d, 0x1e5b, 0xafd3, 0xf757, 0x1bd9, 0x96df, 0x03df, 0x28d6, 0xbb77, 0xd5b5,
            0x03d3, 0xc078, 0x255b, 0xee39, 0x9705, 0x7fcc, 0xf16e, 0x16ca, 0x71d1, 0x9107, 0x00a5,
            0x103d, 0x0b12, 0xea24, 0xdf09, 0x7745, 0x7c1b, 0xcdba, 0x3093, 0x742e, 0x1e4c, 0x087b,
            0x9661, 0x0f3a, 0x6c51, 0xdc63, 0xb9d8, 0xf518, 0x09e1, 0x1426, 0xb6dc, 0xc246, 0xa273,
            0x5562, 0x8fde, 0x8f0e, 0xd034, 0x6651, 0x95ec, 0x6452, 0x95d4, 0xdf84, 0x118c, 0x44ab,
            0x328b, 0xf3d1, 0xb048, 0x2081, 0x748a, 0x05ee, 0x0f9b, 0x8110, 0x46e8, 0x6476, 0x8863,
            0x9850, 0xcb94, 0x2d2e, 0xcbac, 0xce53, 0x91bb, 0xa605, 0xfe50, 0x06f5, 0xef2d, 0xbd7c,
            0x736b, 0xf371, 0x6055, 0x6ab9, 0x135f, 0xb572, 0x5eb1, 0x7a36, 0xe4d5, 0xb998, 0xa7ea,
            0x1d06, 0x1275, 0x7f89, 0x3c92, 0xe906, 0x40c1, 0x8207, 0x058e, 0xa660, 0x72cd, 0xce25,
            0xd92a, 0x7731, 0x7633, 0xc6da, 0xb213, 0x0a93, 0x30c0, 0x58d3, 0x5ac0, 0x3ce7, 0x1028,
            0x4bcd, 0x86b9, 0x7f60, 0x22a6, 0x0ce9, 0xb569, 0x8c83, 0xb5bf, 0x2dd9, 0x7bdd, 0xc4bc,
            0xce57, 0x0b0b, 0x0a9c, 0xd74a, 0x6936, 0x0e40, 0xa874, 0x02b2, 0xfe8d, 0x0c16, 0xa0e0,
            0x5b01, 0x6f18, 0x6264, 0x4e77, 0x01a0, 0x3484, 0xe5b4, 0xf0cc, 0xd30d, 0x7904, 0x8216,
            0x46dd, 0x6fc0, 0xfa77, 0x8c3e, 0x5c10, 0xf776, 0x3043, 0x23dc, 0xfffc, 0x35c0, 0x8007,
            0x7993, 0xf198, 0x94eb, 0xe9bf, 0x7cc0, 0x170d, 0xea0d, 0xa7d0, 0x3d77, 0x7d6e, 0xc8f7,
            0x9a86, 0x6462, 0xc8d2, 0x357a, 0x8fa0, 0xf201, 0x55e5, 0x5235, 0x7da1, 0x52e6, 0xcc31,
            0xbecd, 0x3343, 0x343a, 0x2b1f, 0xd19e, 0x4cc6, 0x83a2, 0x6d16, 0x9c97, 0xa61b, 0xde54,
            0x6da1, 0xa57e, 0x44a7, 0x1e84, 0x98e7, 0x0e44, 0x5494, 0xe013, 0x0ed2, 0x0b3a, 0xa2db,
            0xc93a, 0xe6a0, 0xdccd, 0x84ac, 0xc898, 0xb974, 0x3d62, 0xe4cf, 0xcbc3, 0xa7bd, 0xde59,
            0x9391, 0x5635, 0xdac1, 0xd9b6, 0x1700, 0x7b35, 0x9555, 0x648e, 0xdacd, 0xffdf, 0xdd6a,
            0x9616, 0xea2e, 0xb1a4, 0x80c1, 0xdb21, 0x1076, 0x9543, 0xc165, 0x66d8, 0x26b8, 0x7095,
            0xdf4f, 0xcf4b, 0x1cec, 0xb231, 0x4037, 0x9fa5, 0x3637, 0xf96e, 0x215a, 0x65c9, 0x4696,
            0x734a, 0x556e, 0xb47f, 0x5160, 0xbf85, 0x850b, 0x06e0, 0x8181, 0x45f7, 0x202b, 0x86d1,
            0x5de7, 0x8ecd, 0xf77c, 0x031f, 0xa330, 0x79b4, 0xf38b, 0x59a8, 0x68cf, 0xf885, 0xfc87,
            0x4054, 0xe627, 0x845e, 0xa77f, 0x8450, 0x2302, 0x86e6, 0x2d94, 0xbbf7, 0x9e54, 0x2d79,
            0x1aa6, 0x6c50, 0xaef5, 0xbd9d, 0x85f3, 0x7b05, 0x5ec3, 0x6d70, 0x3ff3, 0x62a6, 0x252a,
            0x72c4, 0x2f56, 0xf9c1, 0xadf9, 0x00ff, 0xedfc, 0xddf3, 0x439c, 0x2777, 0xb742, 0xddfd,
            0x14fc, 0xa147, 0xd950, 0x37bd, 0x6296, 0xf816, 0x29af, 0x297c, 0xbf24, 0x6f05, 0xe8a4,
            0x17f4, 0xc8ab, 0xc0d1, 0x87b2, 0xeca2, 0x1b31, 0xa20b, 0xaad8, 0xd46c, 0x636f, 0x3975,
            0x363e, 0xdc79, 0xc450, 0x507e, 0xd8d5, 0x74c9, 0x56de, 0x92bc, 0x05eb, 0x749a, 0x3d98,
            0xf26a, 0x23fe, 0x4f29, 0x7856, 0x968c, 0x8794, 0x2835, 0x8dc3, 0xa440, 0x3b7b, 0xcc28,
            0x98e6, 0x36f1, 0xf305, 0x7641, 0xe895, 0x88d7, 0xedb3, 0x934a, 0x88c2, 0x0d19, 0xd558,
            0xe4bd, 0xe365, 0x5b52, 0xd26d, 0x77be, 0xe2cc, 0xd759, 0xb890, 0x5924, 0xf681, 0xfd5f,
            0xccf7, 0xc9b7, 0x544a, 0x1fe8, 0xacd1, 0x349e, 0xf889, 0x3e38, 0x980a, 0xfcf6, 0x4aaf,
            0xc970, 0x2699, 0xce48, 0x3229, 0x148e, 0x2c20, 0x28c1, 0x7fc3, 0x1cf6, 0x080c, 0x2f85,
            0x6ed0, 0xa884, 0xd958, 0xd555, 0x480d, 0x8874, 0xe8d4, 0x7c66, 0x226f, 0xbf4f, 0xbcea,
            0x3eeb, 0xac04, 0xc774, 0xbc95, 0xa97f, 0x8382, 0x165b, 0xc178, 0x708e, 0x8be5, 0x7eb4,
            0x84ad, 0x15d5, 0x5193, 0x4114, 0xd320, 0x9add, 0x85a3, 0x8b70, 0x1be3, 0xa39d, 0xbf82,
            0x6e04, 0x3bd2, 0xdf31, 0x0741, 0xaab8, 0xd398, 0x01f4, 0xdd3a, 0x2f9d, 0x2b55, 0x6811,
            0x171f,
        ]);
        let expected_world_checksum = Checksum([
            171, 53, 185, 10, 179, 49, 48, 151, 87, 43, 141, 13, 43, 152, 121, 1, 144, 7, 120, 188,
            115, 248, 214, 220, 229, 210, 175, 134, 215, 231, 18, 245,
        ]);

        for (input, expected_lt_hash, expected_checksum) in [
            ("hello", expected_hello_lt_hash, expected_hello_checksum),
            ("world!", expected_world_lt_hash, expected_world_checksum),
        ] {
            let mut hasher = blake3::Hasher::new();
            hasher.update(input.as_bytes());
            let actual_lt_hash = LtHash::with(&hasher);
            assert_eq!(actual_lt_hash, expected_lt_hash);
            let actual_checksum = actual_lt_hash.checksum();
            assert_eq!(actual_checksum, expected_checksum);
        }
    }
}
