use rstar::{RTreeObject, AABB};

/// Distinguishes whether an agent belongs to an Organism or a SemiLife entity.
///
/// The `organism_id` field in [`Agent`] serves as a local array index for whichever
/// type owns the agent — disambiguation is only needed in code that iterates all agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnerType {
    Organism,
    SemiLife,
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: u32,
    /// Local index into the owning entity's runtime array (organisms or semi_lives).
    pub organism_id: u16,
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    pub internal_state: [f32; 4],
    /// Whether this agent belongs to an `OrganismRuntime` or `SemiLifeRuntime`.
    pub owner_type: OwnerType,
}

impl Agent {
    /// Create an agent owned by an [`crate::organism::OrganismRuntime`].
    pub fn new(id: u32, organism_id: u16, position: [f64; 2]) -> Self {
        Self {
            id,
            organism_id,
            position,
            velocity: [0.0; 2],
            internal_state: [0.5; 4],
            owner_type: OwnerType::Organism,
        }
    }

    /// Create an agent owned by a [`crate::semi_life::SemiLifeRuntime`].
    pub fn for_semi_life(id: u32, semi_life_id: u16, position: [f64; 2]) -> Self {
        Self {
            id,
            organism_id: semi_life_id,
            position,
            velocity: [0.0; 2],
            internal_state: [0.5; 4],
            owner_type: OwnerType::SemiLife,
        }
    }
}

impl RTreeObject for Agent {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.position)
    }
}
