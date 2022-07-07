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
            Instruction::SkipEqI(i, x, y) => format!("{:#06x} skip if V{:1X} == {:#04x}", i, x, y),
            Instruction::SkipNeqI(i, x, y) => format!("{:#06x} skip if V{:1X} != {:#04x}", i, x, y),
            Instruction::SkipEq(i, x, y) => format!("{:#06x} skip if V{:1X} == V{:1X}", i, x, y),
            Instruction::SkipNeq(i, x, y) => format!("{:#06x} skip if V{:1X} != V{:1X}", i, x, y),
            Instruction::Set(i, x, y) => format!("{:#06x} set V{:1X} = {:#04x}", i, x, y),
            Instruction::Add(i, x, nn) => format!("{:#06x} add {:#04x} to V{:1X}", i, nn, x),
            Instruction::Arithmetic(i, op) => format!("{:#06x} {}", i, op),
            Instruction::SetIdx(i, addr) => format!("{:#06x} set I to {:#05x}", i, addr),
            Instruction::JumpX(i, addr) => format!("{:#06x} jump to {:#05x} + V0", i, addr),
            Instruction::Random(i, x, nn) => format!("{:#06x} set V{:1X} to random with mask {:#04x}", i, x, nn),
            Instruction::Display { i, x, y, n } => {
                format!("{:#06x} draw 8-by-{} sprite at (V{:1X}, V{:1X})", i, n, x, y)
            }
            Instruction::SkipOnKey(i, x) => format!("{:#06x} skip if V{:1X} key pressed", i, x),
            Instruction::SkipOffKey(i, x) => format!("{:#06x} skip if V{:1X} key not pressed", i, x),
            Instruction::ReadDelay(i, x) => format!("{:#06x} read delay timer into V{:1X}", i, x),
            Instruction::SetDelay(i, x) => format!("{:#06x} set delay timer to V{:1X}", i, x),
            Instruction::SetSound(i, x) => format!("{:#06x} set sound timer to V{:1X}", i, x),
            Instruction::AddIdx(i, x) => format!("{:#06x} add V{:1X} to I", i, x),
            Instruction::GetKey(i, x) => format!("{:#06x} wait for key press and store in V{:1X}", i, x),
            Instruction::FontChar(i, x) => format!("{:#06x} set I to font sprite for char V{:1X}", i, x),
            Instruction::BCD(i, x) => format!("{:#06x} set mem @ I to BCD of V{:1X}", i, x),
            Instruction::StoreMem(i, x) => format!("{:#06x} set mem @ I to V0-V{:1X}", i, x),
            Instruction::ReadMem(i, x) => format!("{:#06x} read mem @ I to V0-V{:1X}", i, x),
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

#[cfg(test)]
mod test_instruction {
    use crate::{Instruction, Operation};

    #[test]
    fn decode() {
        let tests = vec![
            (0x00e0, Instruction::Clear(0x00e0)),
            (0x1abc, Instruction::Jump(0x1abc, 0xabc)),
            (0x2abc, Instruction::SubroutinePush(0x2abc, 0xabc)),
            (0x00ee, Instruction::SubroutinePop(0x00ee)),
            (0x3abc, Instruction::SkipEqI(0x3abc, 0xa, 0xbc)),
            (0x4abc, Instruction::SkipNeqI(0x4abc, 0xa, 0xbc)),
            (0x5ab0, Instruction::SkipEq(0x5ab0, 0xa, 0xb)),
            (0x9ab0, Instruction::SkipNeq(0x9ab0, 0xa, 0xb)),
            (0x6abc, Instruction::Set(0x6abc, 0xa, 0xbc)),
            (0x7abc, Instruction::Add(0x7abc, 0xa, 0xbc)),
            (0x8ab0, Instruction::Arithmetic(0x8ab0, Operation::Set(0xa, 0xb))),
            (0x8ab1, Instruction::Arithmetic(0x8ab1, Operation::Or(0xa, 0xb))),
            (0x8ab2, Instruction::Arithmetic(0x8ab2, Operation::And(0xa, 0xb))),
            (0x8ab3, Instruction::Arithmetic(0x8ab3, Operation::Xor(0xA, 0xb))),
            (0x8ab4, Instruction::Arithmetic(0x8ab4, Operation::Add(0xa, 0xb))),
            (0x8ab5, Instruction::Arithmetic(0x8ab5, Operation::Sub(0xa, 0xb))),
            (0x8ab7, Instruction::Arithmetic(0x8ab7, Operation::Sub(0xb, 0xa))),
            (0x8ab6, Instruction::Arithmetic(0x8ab6, Operation::ShiftR(0xa, 0xb))),
            (0x8abe, Instruction::Arithmetic(0x8abe, Operation::ShiftL(0xa, 0xb))),
            (0xaabc, Instruction::SetIdx(0xaabc, 0xabc)),
            (0xbabc, Instruction::JumpX(0xbabc, 0xabc)),
            (0xcabc, Instruction::Random(0xcabc, 0xa, 0xbc)),
            (
                0xdabc,
                Instruction::Display {
                    i: 0xdabc,
                    x: 0xa,
                    y: 0xb,
                    n: 0xc,
                },
            ),
            (0xea9e, Instruction::SkipOnKey(0xea9e, 0xa)),
            (0xeaa1, Instruction::SkipOffKey(0xeaa1, 0xa)),
            (0xfa07, Instruction::ReadDelay(0xfa07, 0xa)),
            (0xfa15, Instruction::SetDelay(0xfa15, 0xa)),
            (0xfa18, Instruction::SetSound(0xfa18, 0xa)),
            (0xfa1e, Instruction::AddIdx(0xfa1e, 0xa)),
            (0xfa0a, Instruction::GetKey(0xfa0a, 0xa)),
            (0xfa29, Instruction::FontChar(0xfa29, 0xa)),
            (0xfa33, Instruction::BCD(0xfa33, 0xa)),
            (0xfa55, Instruction::StoreMem(0xfa55, 0xa)),
            (0xfa65, Instruction::ReadMem(0xfa65, 0xa)),
        ];
        for test in tests {
            assert_eq!(Instruction::from(test.0), test.1);
        }
    }
}
