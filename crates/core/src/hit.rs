use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub dist: f32,
    pub id: u32,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == Ordering::Equal
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}
