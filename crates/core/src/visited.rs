pub struct Visited {
    stamps: Vec<u32>,
    gen: u32,
}

impl Default for Visited {
    fn default() -> Self {
        Self {
            stamps: Vec::new(),
            gen: 1,
        }
    }
}

impl Visited {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            stamps: vec![0; n],
            gen: 1,
        }
    }

    pub fn grow(&mut self, n: usize) {
        let len = self.stamps.len();
        if n > len {
            self.stamps.resize(n.max(len * 2), 0);
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        if self.gen == u32::MAX {
            self.stamps.fill(0);
            self.gen = 1;
        } else {
            self.gen += 1;
        }
    }

    #[inline]
    pub fn mark(&mut self, i: usize) -> bool {
        let g = self.gen;
        let s = &mut self.stamps[i];
        if *s == g {
            false
        } else {
            *s = g;
            true
        }
    }
}
