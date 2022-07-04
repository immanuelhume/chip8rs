use std::fmt;

/// A Chip 8 instruction. The first item in each tuple is the numerical representation of the
/// instruction.
#[derive(Debug, PartialEq)]
pub enum Instruction {
    Nop(u16),   // just to be safe
    Clear(u16), // 0x00E0

    Jump(u16, u16),           // 0x1NNN
    SubroutinePush(u16, u16), // 0x2NNN
    SubroutinePop(u16),       // 0x00EE

    SkipEqI(u16, u8, u8),  // 0x3XNN
    SkipNeqI(u16, u8, u8), // 0x4XNN
    SkipEq(u16, u8, u8),   // 0x5XY0
    SkipNeq(u16, u8, u8),  // 0x9XY0

    Set(u16, u8, u8), // 0x6XNN
    Add(u16, u8, u8), // 0x7XNN

    Arithmetic(u16, Operation), // a bunch

    SetIdx(u16, u16),                        // 0xANNN
    JumpX(u16, u16),                         // 0xBNNN
    Random(u16, u8, u8),                     // 0xCXNN
    Display { i: u16, x: u8, y: u8, n: u8 }, // 0xDXYN

    SkipOnKey(u16, u8),  // 0xEX9E
    SkipOffKey(u16, u8), // 0xEXA1

    ReadDelay(u16, u8), // 0xFX07
    SetDelay(u16, u8),  // 0xFX15
    SetSound(u16, u8),  // 0xFX18

    AddIdx(u16, u8),   // 0xFX1E
    GetKey(u16, u8),   // 0xFX0A
    FontChar(u16, u8), // 0xFX29
    BCD(u16, u8),      // 0xFX33

    StoreMem(u16, u8), // 0xFX55
    ReadMem(u16, u8),  // 0xFX65
}

impl Instruction {
    pub fn from(i: u16) -> Self {
        let opcode = (i & 0xF000) >> 12;
        let x = ((i & 0x0F00) >> 8) as u8;
        let y = ((i & 0x00F0) >> 4) as u8;
        let n = (i & 0x000F) as u8;
        let nn = (i & 0x00FF) as u8;
        let nnn = (i & 0x0FFF) as u16;

        match opcode {
            0x0 => match nnn {
                0x0E0 => Instruction::Clear(i),
                0x0EE => Instruction::SubroutinePop(i),
                _ => Instruction::Nop(i),
            },
            0x1 => Instruction::Jump(i, nnn),
            0x2 => Instruction::SubroutinePush(i, nnn),
            0x3 => Instruction::SkipEqI(i, x, nn),
            0x4 => Instruction::SkipNeqI(i, x, nn),
            0x5 => Instruction::SkipEq(i, x, y),
            0x6 => Instruction::Set(i, x, nn),
            0x7 => Instruction::Add(i, x, nn),
            0x8 => match n {
                0x0 => Instruction::Arithmetic(i, Operation::Set(x, y)),
                0x1 => Instruction::Arithmetic(i, Operation::Or(x, y)),
                0x2 => Instruction::Arithmetic(i, Operation::And(x, y)),
                0x3 => Instruction::Arithmetic(i, Operation::Xor(x, y)),
                0x4 => Instruction::Arithmetic(i, Operation::Add(x, y)),
                0x5 => Instruction::Arithmetic(i, Operation::Sub(x, y)),
                0x7 => Instruction::Arithmetic(i, Operation::Sub(y, x)),
                0x6 => Instruction::Arithmetic(i, Operation::ShiftR(x, y)),
                0xE => Instruction::Arithmetic(i, Operation::ShiftL(x, y)),
                _ => Instruction::Nop(i),
            },
            0x9 => Instruction::SkipNeq(i, x, y),
            0xA => Instruction::SetIdx(i, nnn),
            0xB => Instruction::JumpX(i, nnn),
            0xC => Instruction::Random(i, x, nn),
            0xD => Instruction::Display { i, x, y, n },
            0xE => match nn {
                0x9E => Instruction::SkipOnKey(i, x),
                0xA1 => Instruction::SkipOffKey(i, x),
                _ => Instruction::Nop(i),
            },
            0xF => match nn {
                0x07 => Instruction::ReadDelay(i, x),
                0x15 => Instruction::SetDelay(i, x),
                0x18 => Instruction::SetSound(i, x),
                0x1E => Instruction::AddIdx(i, x),
                0x0A => Instruction::GetKey(i, x),
                0x29 => Instruction::FontChar(i, x),
                0x33 => Instruction::BCD(i, x),
                0x55 => Instruction::StoreMem(i, x),
                0x65 => Instruction::ReadMem(i, x),
                _ => Instruction::Nop(i),
            },
            _ => Instruction::Nop(i),
        }
    }
}

impl ToString for Instruction {
    fn to_string(&self) -> String {
        match self {
            Instruction::Nop(i) => format!("{:#06x} nop", i),
            Instruction::Clear(i) => format!("{:#06x} clear screen", i),
            Instruction::Jump(i, addr) => format!("{:#06x} jump to {:#05x}", i, addr),
            Instruction::SubroutinePush(i, addr) => format!("{:#06x} subroutine at {:#06x}", i, addr),
            Instruction::SubroutinePop(i) => format!("{:#06x} return from subroutine", i),
            Instruction::SkipEqI(i, x, y) => format!("{:#06x} skip if V{} == {:#04x}", i, x, y),
            Instruction::SkipNeqI(i, x, y) => format!("{:#06x} skip if V{} != {:#04x}", i, x, y),
            Instruction::SkipEq(i, x, y) => format!("{:#06x} skip if V{} == V{}", i, x, y),
            Instruction::SkipNeq(i, x, y) => format!("{:#06x} skip if V{} != V{}", i, x, y),
            Instruction::Set(i, x, y) => format!("{:#06x} set V{} = {:#04x}", i, x, y),
            Instruction::Add(i, x, nn) => format!("{:#06x} add {:#04x} to V{}", i, nn, x),
            Instruction::Arithmetic(i, op) => format!("{:#06x} {}", i, op),
            Instruction::SetIdx(i, addr) => format!("{:#06x} set I to {:#05x}", i, addr),
            Instruction::JumpX(i, addr) => format!("{:#06x} jump to {:#05x} + V0", i, addr),
            Instruction::Random(i, x, nn) => format!("{:#06x} set V{} to random with mask {:#04x}", i, x, nn),
            Instruction::Display { i, x, y, n } => {
                format!("{:#06x} draw 8-by-{} sprite at (V{}, V{})", i, n, x, y)
            }
            Instruction::SkipOnKey(i, x) => format!("{:#06x} skip if V{} key pressed", i, x),
            Instruction::SkipOffKey(i, x) => format!("{:#06x} skip if V{} key not pressed", i, x),
            Instruction::ReadDelay(i, x) => format!("{:#06x} read delay timer into V{}", i, x),
            Instruction::SetDelay(i, x) => format!("{:#06x} set delay timer to V{}", i, x),
            Instruction::SetSound(i, x) => format!("{:#06x} set sound timer to V{}", i, x),
            Instruction::AddIdx(i, x) => format!("{:#06x} add V{} to I", i, x),
            Instruction::GetKey(i, x) => format!("{:#06x} wait for key press and store in V{}", i, x),
            Instruction::FontChar(i, x) => format!("{:#06x} set I to font sprite for char V{}", i, x),
            Instruction::BCD(i, x) => format!("{:#06x} set mem @ I to BCD of V{}", i, x),
            Instruction::StoreMem(i, x) => format!("{:#06x} set mem @ I to V0-V{}", i, x),
            Instruction::ReadMem(i, x) => format!("{:#06x} read mem @ I to V0-V{}", i, x),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Operation {
    Set(u8, u8),    // 0x8XY0
    Or(u8, u8),     // 0x8XY1
    And(u8, u8),    // 0x8XY2
    Xor(u8, u8),    // 0x8XY3
    Add(u8, u8),    // 0x8XY4
    Sub(u8, u8),    // 0x8XY5, 0x8XY7
    ShiftL(u8, u8), // 0x8XY6
    ShiftR(u8, u8), // 0x8XYE
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operation::Set(x, y) => write!(f, "set V{} = V{}", x, y),
            Operation::Or(x, y) => write!(f, "V{} = V{} | V{}", x, x, y),
            Operation::And(x, y) => write!(f, "V{} = V{} & V{}", x, x, y),
            Operation::Xor(x, y) => write!(f, "V{} = V{} ^ V{}", x, x, y),
            Operation::Add(x, y) => write!(f, "V{} = V{} + V{}", x, x, y),
            Operation::Sub(x, y) => write!(f, "V{} = V{} - V{}", x, x, y),
            Operation::ShiftL(x, y) => write!(f, "V{} = V{} << 1", x, y),
            Operation::ShiftR(x, y) => write!(f, "V{} = V{} >> 1", x, y),
        }
    }
}
