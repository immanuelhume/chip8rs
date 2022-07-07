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
        use Instruction::*;

        let opcode = (i & 0xF000) >> 12;
        let x = ((i & 0x0F00) >> 8) as u8;
        let y = ((i & 0x00F0) >> 4) as u8;
        let n = (i & 0x000F) as u8;
        let nn = (i & 0x00FF) as u8;
        let nnn = (i & 0x0FFF) as u16;

        match opcode {
            0x0 => match nnn {
                0x0E0 => Clear(i),
                0x0EE => SubroutinePop(i),
                _ => Nop(i),
            },
            0x1 => Jump(i, nnn),
            0x2 => SubroutinePush(i, nnn),
            0x3 => SkipEqI(i, x, nn),
            0x4 => SkipNeqI(i, x, nn),
            0x5 => SkipEq(i, x, y),
            0x6 => Set(i, x, nn),
            0x7 => Add(i, x, nn),
            0x8 => match n {
                0x0 => Arithmetic(i, Operation::Set(x, y)),
                0x1 => Arithmetic(i, Operation::Or(x, y)),
                0x2 => Arithmetic(i, Operation::And(x, y)),
                0x3 => Arithmetic(i, Operation::Xor(x, y)),
                0x4 => Arithmetic(i, Operation::Add(x, y)),
                0x5 => Arithmetic(i, Operation::Sub(x, y)),
                0x7 => Arithmetic(i, Operation::Sub(y, x)),
                0x6 => Arithmetic(i, Operation::ShiftR(x, y)),
                0xE => Arithmetic(i, Operation::ShiftL(x, y)),
                _ => Nop(i),
            },
            0x9 => SkipNeq(i, x, y),
            0xA => SetIdx(i, nnn),
            0xB => JumpX(i, nnn),
            0xC => Random(i, x, nn),
            0xD => Display { i, x, y, n },
            0xE => match nn {
                0x9E => SkipOnKey(i, x),
                0xA1 => SkipOffKey(i, x),
                _ => Nop(i),
            },
            0xF => match nn {
                0x07 => ReadDelay(i, x),
                0x15 => SetDelay(i, x),
                0x18 => SetSound(i, x),
                0x1E => AddIdx(i, x),
                0x0A => GetKey(i, x),
                0x29 => FontChar(i, x),
                0x33 => BCD(i, x),
                0x55 => StoreMem(i, x),
                0x65 => ReadMem(i, x),
                _ => Nop(i),
            },
            _ => Nop(i),
        }
    }
}

impl ToString for Instruction {
    fn to_string(&self) -> String {
        use Instruction::*;
        match self {
            Nop(i) => format!("{:#06x} nop", i),
            Clear(i) => format!("{:#06x} clear screen", i),
            Jump(i, addr) => format!("{:#06x} jump to {:#05x}", i, addr),
            SubroutinePush(i, addr) => format!("{:#06x} subroutine at {:#06x}", i, addr),
            SubroutinePop(i) => format!("{:#06x} return from subroutine", i),
            SkipEqI(i, x, y) => format!("{:#06x} skip if V{:1X} == {:#04x}", i, x, y),
            SkipNeqI(i, x, y) => format!("{:#06x} skip if V{:1X} != {:#04x}", i, x, y),
            SkipEq(i, x, y) => format!("{:#06x} skip if V{:1X} == V{:1X}", i, x, y),
            SkipNeq(i, x, y) => format!("{:#06x} skip if V{:1X} != V{:1X}", i, x, y),
            Set(i, x, y) => format!("{:#06x} set V{:1X} = {:#04x}", i, x, y),
            Add(i, x, nn) => format!("{:#06x} add {:#04x} to V{:1X}", i, nn, x),
            Arithmetic(i, op) => format!("{:#06x} {}", i, op),
            SetIdx(i, addr) => format!("{:#06x} set I to {:#05x}", i, addr),
            JumpX(i, addr) => format!("{:#06x} jump to {:#05x} + V0", i, addr),
            Random(i, x, nn) => format!("{:#06x} set V{:1X} to random with mask {:#04x}", i, x, nn),
            Display { i, x, y, n } => format!("{:#06x} draw 8-by-{} sprite at (V{:1X}, V{:1X})", i, n, x, y),
            SkipOnKey(i, x) => format!("{:#06x} skip if V{:1X} key pressed", i, x),
            SkipOffKey(i, x) => format!("{:#06x} skip if V{:1X} key not pressed", i, x),
            ReadDelay(i, x) => format!("{:#06x} read delay timer into V{:1X}", i, x),
            SetDelay(i, x) => format!("{:#06x} set delay timer to V{:1X}", i, x),
            SetSound(i, x) => format!("{:#06x} set sound timer to V{:1X}", i, x),
            AddIdx(i, x) => format!("{:#06x} add V{:1X} to I", i, x),
            GetKey(i, x) => format!("{:#06x} wait for key press and store in V{:1X}", i, x),
            FontChar(i, x) => format!("{:#06x} set I to font sprite for char V{:1X}", i, x),
            BCD(i, x) => format!("{:#06x} set mem @ I to BCD of V{:1X}", i, x),
            StoreMem(i, x) => format!("{:#06x} set mem @ I to V0-V{:1X}", i, x),
            ReadMem(i, x) => format!("{:#06x} read mem @ I to V0-V{:1X}", i, x),
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
        use Operation::*;
        match self {
            Set(x, y) => write!(f, "set V{} = V{}", x, y),
            Or(x, y) => write!(f, "V{} = V{} | V{}", x, x, y),
            And(x, y) => write!(f, "V{} = V{} & V{}", x, x, y),
            Xor(x, y) => write!(f, "V{} = V{} ^ V{}", x, x, y),
            Add(x, y) => write!(f, "V{} = V{} + V{}", x, x, y),
            Sub(x, y) => write!(f, "V{} = V{} - V{}", x, x, y),
            ShiftL(x, y) => write!(f, "V{} = V{} << 1", x, y),
            ShiftR(x, y) => write!(f, "V{} = V{} >> 1", x, y),
        }
    }
}

#[cfg(test)]
mod test_instruction {
    use crate::{
        Instruction::{self, *},
        Operation,
    };

    #[test]
    fn decode() {
        let tests = vec![
            (0x00e0, Clear(0x00e0)),
            (0x1abc, Jump(0x1abc, 0xabc)),
            (0x2abc, SubroutinePush(0x2abc, 0xabc)),
            (0x00ee, SubroutinePop(0x00ee)),
            (0x3abc, SkipEqI(0x3abc, 0xa, 0xbc)),
            (0x4abc, SkipNeqI(0x4abc, 0xa, 0xbc)),
            (0x5ab0, SkipEq(0x5ab0, 0xa, 0xb)),
            (0x9ab0, SkipNeq(0x9ab0, 0xa, 0xb)),
            (0x6abc, Set(0x6abc, 0xa, 0xbc)),
            (0x7abc, Add(0x7abc, 0xa, 0xbc)),
            (0x8ab0, Arithmetic(0x8ab0, Operation::Set(0xa, 0xb))),
            (0x8ab1, Arithmetic(0x8ab1, Operation::Or(0xa, 0xb))),
            (0x8ab2, Arithmetic(0x8ab2, Operation::And(0xa, 0xb))),
            (0x8ab3, Arithmetic(0x8ab3, Operation::Xor(0xA, 0xb))),
            (0x8ab4, Arithmetic(0x8ab4, Operation::Add(0xa, 0xb))),
            (0x8ab5, Arithmetic(0x8ab5, Operation::Sub(0xa, 0xb))),
            (0x8ab7, Arithmetic(0x8ab7, Operation::Sub(0xb, 0xa))),
            (0x8ab6, Arithmetic(0x8ab6, Operation::ShiftR(0xa, 0xb))),
            (0x8abe, Arithmetic(0x8abe, Operation::ShiftL(0xa, 0xb))),
            (0xaabc, SetIdx(0xaabc, 0xabc)),
            (0xbabc, JumpX(0xbabc, 0xabc)),
            (0xcabc, Random(0xcabc, 0xa, 0xbc)),
            (
                0xdabc,
                Display {
                    i: 0xdabc,
                    x: 0xa,
                    y: 0xb,
                    n: 0xc,
                },
            ),
            (0xea9e, SkipOnKey(0xea9e, 0xa)),
            (0xeaa1, SkipOffKey(0xeaa1, 0xa)),
            (0xfa07, ReadDelay(0xfa07, 0xa)),
            (0xfa15, SetDelay(0xfa15, 0xa)),
            (0xfa18, SetSound(0xfa18, 0xa)),
            (0xfa1e, AddIdx(0xfa1e, 0xa)),
            (0xfa0a, GetKey(0xfa0a, 0xa)),
            (0xfa29, FontChar(0xfa29, 0xa)),
            (0xfa33, BCD(0xfa33, 0xa)),
            (0xfa55, StoreMem(0xfa55, 0xa)),
            (0xfa65, ReadMem(0xfa65, 0xa)),
        ];
        for test in tests {
            assert_eq!(Instruction::from(test.0), test.1);
        }
    }
}
