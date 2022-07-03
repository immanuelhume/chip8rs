use crossterm::{
    self,
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    env, error, fmt,
    io::{self, Read},
    sync::{
        mpsc::{self, TryRecvError},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time,
};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{self, Constraint, Layout},
    style::Color,
    style::{Modifier, Style},
    symbols::Marker,
    text::{Span, Spans, Text},
    widgets::{
        canvas::{Canvas, Points},
        Block, Borders, List, ListItem, ListState,
    },
    Frame, Terminal,
};

fn main() -> Result<(), io::Error> {
    // Read the entire CHIP-8 file into memory.
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    let mut rom = vec![];
    {
        let mut file = std::fs::File::open(filename)?;
        file.read_to_end(&mut rom)?;
    }

    // Set up emulator and UI.
    let (tx, rx) = mpsc::channel();
    let mut emulator = Emulator::new(tx, time::Duration::from_millis(10));
    emulator.load(&rom).expect("could not load ROM");
    let cmds = rom
        .iter()
        .fold(vec![], |mut accum, byte| {
            accum.push((0 | *byte as u16) << 8 | *byte as u16);
            accum
        })
        .iter()
        .map(|instruction| Command::from(*instruction))
        .collect();
    let mut ui = UI::new(rx, emulator.screen.pixels.clone(), cmds);

    thread::spawn(move || {
        emulator.run().expect("emulator crashed");
    });
    ui.run();

    Ok(())
}

struct Emulator {
    pc: usize,
    mem: [u8; 0x1000],
    updates: mpsc::Sender<Update>,
    poll_timeout: time::Duration,
    screen: Screen,
    idx_reg: usize,
    registers: [u8; 0x10],
    stack: Vec<usize>,
    delay_timer: Timer,
    sound_timer: Timer,
}

impl Emulator {
    fn new(tx: mpsc::Sender<Update>, poll_timeout: time::Duration) -> Self {
        let mut mem = [0; 0x1000];
        mem.write(0x100, &FONT_DATA).unwrap(); // Write font sprits to 0x100.
        Self {
            pc: 0x200,
            mem,
            poll_timeout,
            updates: tx.clone(),
            screen: Screen::new(tx),
            idx_reg: 0,
            registers: [0; 0x10],
            stack: vec![],
            delay_timer: Timer::new(0, TIMER_INTERVAL),
            sound_timer: Timer::new(0, TIMER_INTERVAL),
        }
    }

    fn load(&mut self, rom: &[u8]) -> Result<(), ErrMemory> {
        if rom.len() > 0x1000 - 0x200 {
            Err(ErrMemory::Overflow)
        } else {
            self.mem[0x200..0x200 + rom.len()].copy_from_slice(rom);
            Ok(())
        }
    }

    fn fetch_next_and_decode(&mut self) -> Result<Command, ErrMemory> {
        if let [a, b] = *self.mem.read(self.pc, 2)? {
            self.pc += 2;
            let i: u16 = (0 | a as u16) << 8 | b as u16;
            Ok(Command::from(i))
        } else {
            // This should never happen.
            panic!()
        }
    }

    fn run(&mut self) -> Result<(), Box<dyn error::Error>> {
        loop {
            let next = self.fetch_next_and_decode().expect("could not decode next instruction");
            self.exec(next);

            if event::poll(self.poll_timeout)? {
                if let event::Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('k') => {
                            self.updates.send(Update::Exit)?;
                            break;
                        }
                        KeyCode::Char('p') => loop {
                            if let event::Event::Key(key) = event::read()? {
                                if let KeyCode::Char('p') = key.code {
                                    break;
                                }
                            }
                        },
                        _ => (),
                    }
                }
            }

            thread::sleep(EMULATOR_INTERVAL);
        }

        Ok(())
    }

    fn exec(&mut self, cmd: Command) {
        self.updates.send(Update::ProgramCounter(self.pc)).unwrap();
        match cmd {
            Command::Clear(_) => self.screen.clear(),
            Command::Jump(_, addr) => self.pc = addr,
            Command::SubroutinePush(_, addr) => {
                self.stack.push(self.pc);
                self.pc = addr;
            }
            Command::SubroutinePop(_) => {
                self.pc = self.stack.pop().unwrap();
            }
            Command::SkipEqI(_, reg, val) => {
                if self.registers[reg] == val {
                    self.pc += 2;
                }
            }
            Command::SkipNeqI(_, reg, val) => {
                if self.registers[reg] != val {
                    self.pc += 2;
                }
            }
            Command::SkipEq(_, reg1, reg2) => {
                if self.registers[reg1] == self.registers[reg2] {
                    self.pc += 2;
                }
            }
            Command::SkipNeq(_, reg1, reg2) => {
                if self.registers[reg1] != self.registers[reg2] {
                    self.pc += 2;
                }
            }
            Command::Arithmetic(_, op) => match op {
                Operation::Set(x, y) => self.registers[x] = self.registers[y],
                Operation::Or(x, y) => self.registers[x] |= self.registers[y],
                Operation::And(x, y) => self.registers[x] &= self.registers[y],
                Operation::Xor(x, y) => self.registers[x] ^= self.registers[y],
                Operation::Add(x, y) => {
                    let (res, overflow) = self.registers[x].overflowing_add(self.registers[y]);
                    self.registers[x] = res;
                    if overflow {
                        self.registers[0xF] = 1;
                    }
                }
                Operation::Sub(x, y) => {
                    let (res, overflow) = self.registers[x].overflowing_sub(self.registers[y]);
                    self.registers[x] = res;
                    self.registers[0xF] = if overflow { 0 } else { 1 };
                }
                Operation::ShiftR(x, y) => {
                    let a = self.registers[y];
                    self.registers[0xF] = a & 1;
                    self.registers[x] = a >> 1;
                }
                Operation::ShiftL(x, y) => {
                    let a = self.registers[y];
                    self.registers[0xF] = a & 0x80;
                    self.registers[x] = a << 1;
                }
            },
            Command::Set(_, reg, val) => self.registers[reg] = val,
            Command::Add(_, reg, val) => self.registers[reg] += val,
            Command::SetIdx(_, v) => self.idx_reg = v,
            Command::JumpX(_, addr) => self.pc = self.registers[0] as usize + addr,
            Command::Random(_, reg, val) => self.registers[reg] = rand::random::<u8>() & val,
            Command::Display { x, y, height: n, .. } => {
                let i = self.registers[y] as usize;
                let j = self.registers[x] as usize;
                let sprite = self.mem.read(self.idx_reg, n).unwrap();

                let is_px_turned_off = self.screen.draw_sprite(i, j, sprite);
                self.registers[0xF] = if is_px_turned_off { 1 } else { 0 };
            }
            Command::SkipOnKey(_, x) => {
                if event::poll(self.poll_timeout).unwrap() {
                    if let Event::Key(key) = event::read().unwrap() {
                        if key.code == KeyCode::Char(KEY_MAP[self.registers[x] as usize]) {
                            self.pc += 2;
                        }
                    }
                }
            }
            Command::SkipOffKey(_, x) => {
                if event::poll(self.poll_timeout).unwrap() {
                    if let Event::Key(key) = event::read().unwrap() {
                        if key.code == KeyCode::Char(KEY_MAP[self.registers[x] as usize]) {
                            self.pc -= 2; // dumb trick
                        }
                    }
                }
                self.pc += 2;
            }
            Command::ReadDelay(_, x) => self.registers[x] = self.delay_timer.get(),
            Command::SetDelay(_, x) => self.delay_timer.set(self.registers[x]),
            Command::SetSound(_, x) => self.sound_timer.set(self.registers[x]),
            Command::AddIdx(_, x) => {
                let a = self.idx_reg;
                let sum = a + self.registers[x] as usize;
                if sum > 0x0FFF && a <= 0x0FFF {
                    self.registers[0xF] = 1;
                }
                self.idx_reg = sum;
            }
            Command::GetKey(_, x) => loop {
                if let Event::Key(key) = event::read().unwrap() {
                    if let KeyCode::Char(c) = key.code {
                        match KEY_MAP.iter().position(|x| *x == c) {
                            Some(i) => {
                                self.registers[x] = i as u8;
                                break;
                            }
                            _ => (),
                        }
                    }
                }
            },
            Command::FontChar(_, x) => self.idx_reg = 0x100 + (self.registers[x] & 0x0F) as usize,
            Command::BCD(_, x) => {
                let mut a = self.registers[x];
                for i in 0..3 {
                    self.mem.write(self.idx_reg + i, &[a % 10]).unwrap();
                    a /= 10;
                }
            }
            Command::StoreMem(_, x) => self.mem.write(self.idx_reg, &self.registers[0..=x]).unwrap(),
            Command::ReadMem(_, x) => {
                self.registers[0..=x].copy_from_slice(&self.mem.read(self.idx_reg, x + 1).unwrap())
            }
            _ => (),
        }
    }
}

/// Maps a hexadecimal digit (index in the array) to the corresponding keycode.
const KEY_MAP: [char; 16] = [
    'x', '1', '2', '3', 'q', 'w', 'e', 'a', 's', 'd', 'z', 'c', '4', 'r', 'f', 'v',
];

#[cfg(test)]
mod test_emulator {
    use crate::{split_u16_into_bytes, Command, Emulator, ErrMemory};
    use std::sync::mpsc;
    use std::time;

    /// Convenience method to initalize things for testing.
    fn init() -> (Emulator, Vec<u16>, Vec<u8>) {
        let (tx, _) = mpsc::channel();
        let emu = Emulator::new(tx, time::Duration::MAX);

        // Create some dummy instructions.
        let instructions: Vec<u16> = vec![0x00E0, 0x61AB, 0x7101, 0xA123];
        let instructions_in_bytes: Vec<u8> = instructions
            .iter()
            .flat_map(|x| {
                let (a, b) = split_u16_into_bytes(*x);
                vec![a, b]
            })
            .collect();
        (emu, instructions, instructions_in_bytes)
    }

    #[test]
    fn test_load_rom() {
        let (mut emu, _, bytes) = init();
        emu.load(&bytes[..]).unwrap();
        assert_eq!(&emu.mem[0x200..0x200 + bytes.len()], bytes);

        let too_many = [0; 0x1000];
        let got = emu.load(&too_many);
        assert!(got.is_err());
        assert_eq!(got.unwrap_err(), ErrMemory::Overflow);
    }

    #[test]
    fn fetch_and_decode() {
        let (mut emu, instructions, instructions_in_bytes) = init();
        let test_commands: Vec<Command> = instructions.iter().map(|x| Command::from(*x)).collect();

        emu.load(&instructions_in_bytes[..]).unwrap();

        for cmd in test_commands {
            assert_eq!(emu.fetch_next_and_decode().unwrap(), cmd);
        }
    }
}

fn split_u16_into_bytes(x: u16) -> (u8, u8) {
    (((x & 0xFF00) >> 8) as u8, (x & 0x00FF) as u8)
}

#[test]
fn test_split_u16_into_bytes() {
    let tests = vec![(0xABCD, 0xAB, 0xCD), (0x0000, 0x00, 0x00), (0x0A0A, 0x0A, 0x0A)];
    for test in tests {
        assert_eq!(split_u16_into_bytes(test.0), (test.1, test.2));
    }
}

fn split_u8_into_bits(mut x: u8) -> [bool; 8] {
    let mut res = [false; 8];
    for i in 0..8 {
        res[7 - i] = x & 1 == 1;
        x >>= 1;
    }
    assert_eq!(x, 0);
    res
}

#[test]
fn test_split_u8_into_bits() {
    let tests = vec![(0b01010101, [false, true, false, true, false, true, false, true])];
    for test in tests {
        assert_eq!(split_u8_into_bits(test.0), test.1);
    }
}

#[derive(Debug, PartialEq)]
enum Command {
    Nop(u16),   // just to be safe
    Clear(u16), // 0x00E0

    Jump(u16, usize),           // 0x1NNN
    SubroutinePush(u16, usize), // 0x2NNN
    SubroutinePop(u16),         // 0x00EE

    SkipEqI(u16, usize, u8),    // 0x3XNN
    SkipNeqI(u16, usize, u8),   // 0x4XNN
    SkipEq(u16, usize, usize),  // 0x5XY0
    SkipNeq(u16, usize, usize), // 0x9XY0

    Set(u16, usize, u8), // 0x6XNN
    Add(u16, usize, u8), // 0x7XNN

    Arithmetic(u16, Operation), // a bunch

    SetIdx(u16, usize),     // 0xANNN
    JumpX(u16, usize),      // 0xBNNN
    Random(u16, usize, u8), // 0xCXNN
    Display {
        instruction: u16,
        x: usize,
        y: usize,
        height: usize,
    }, // 0xDXYN

    SkipOnKey(u16, usize),  // 0xEX9E
    SkipOffKey(u16, usize), // 0xEXA1

    ReadDelay(u16, usize), // 0xFX07
    SetDelay(u16, usize),  // 0xFX15
    SetSound(u16, usize),  // 0xFX18

    AddIdx(u16, usize),   // 0xFX1E
    GetKey(u16, usize),   // 0xFX0A
    FontChar(u16, usize), // 0xFX29
    BCD(u16, usize),      // 0xFX33

    StoreMem(u16, usize), // 0xFX55
    ReadMem(u16, usize),  // 0xFX65
}

impl ToString for Command {
    fn to_string(&self) -> String {
        match self {
            Command::Nop(i) => format!("{:#06x} nop", i),
            Command::Clear(i) => format!("{:#06x} clear screen", i),
            Command::Jump(i, addr) => format!("{:#06x} jump to {:#05x}", i, addr),
            Command::SubroutinePush(i, addr) => format!("{:#06x} subroutine at {:#06x}", i, addr),
            Command::SubroutinePop(i) => format!("{:#06x} return from subroutine", i),
            Command::SkipEqI(i, x, y) => format!("{:#06x} skip if V{} == {:#04x}", i, x, y),
            Command::SkipNeqI(i, x, y) => format!("{:#06x} skip if V{} != {:#04x}", i, x, y),
            Command::SkipEq(i, x, y) => format!("{:#06x} skip if V{} == V{}", i, x, y),
            Command::SkipNeq(i, x, y) => format!("{:#06x} skip if V{} != V{}", i, x, y),
            Command::Set(i, x, y) => format!("{:#06x} set V{} = {:#04x}", i, x, y),
            Command::Add(i, x, nn) => format!("{:#06x} add {:#04x} to V{}", i, nn, x),
            Command::Arithmetic(i, op) => format!("{:#06x} {}", i, op),
            Command::SetIdx(i, addr) => format!("{:#06x} set I to {:#05x}", i, addr),
            Command::JumpX(i, addr) => format!("{:#06x} jump to {:#05x} + V0", i, addr),
            Command::Random(i, x, nn) => format!("{:#06x} set V{} to random with mask {:#04x}", i, x, nn),
            Command::Display {
                instruction,
                x,
                y,
                height,
            } => format!("{:#06x} display 8x{} at ({}, {})", instruction, height, x, y),
            Command::SkipOnKey(i, x) => format!("{:#06x} skip if V{} key pressed", i, x),
            Command::SkipOffKey(i, x) => format!("{:#06x} skip if V{} key not pressed", i, x),
            Command::ReadDelay(i, x) => format!("{:#06x} read delay timer into V{}", i, x),
            Command::SetDelay(i, x) => format!("{:#06x} set delay timer to V{}", i, x),
            Command::SetSound(i, x) => format!("{:#06x} set sound timer to V{}", i, x),
            Command::AddIdx(i, x) => format!("{:#06x} add V{} to I", i, x),
            Command::GetKey(i, x) => format!("{:#06x} wait for key press and store in V{}", i, x),
            Command::FontChar(i, x) => format!("{:#06x} set I to font sprite for char V{}", i, x),
            Command::BCD(i, x) => format!("{:#06x} set mem @ I to BCD of V{}", i, x),
            Command::StoreMem(i, x) => format!("{:#06x} set mem @ I to V0-V{}", i, x),
            Command::ReadMem(i, x) => format!("{:#06x} read mem @ I to V0-V{}", i, x),
        }
    }
}

#[derive(Debug, PartialEq)]
enum Operation {
    Set(usize, usize),    // 0x8XY0
    Or(usize, usize),     // 0x8XY1
    And(usize, usize),    // 0x8XY2
    Xor(usize, usize),    // 0x8XY3
    Add(usize, usize),    // 0x8XY4
    Sub(usize, usize),    // 0x8XY5, 0x8XY7
    ShiftL(usize, usize), // 0x8XY6
    ShiftR(usize, usize), // 0x8XYE
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

impl Command {
    fn from(i: u16) -> Self {
        let opcode = (i & 0xF000) >> 12;
        let x = ((i & 0x0F00) >> 8) as usize;
        let y = ((i & 0x00F0) >> 4) as usize;
        let n = (i & 0x000F) as usize;
        let nn = (i & 0x00FF) as u8;
        let nnn = (i & 0x0FFF) as usize;

        match opcode {
            0x0 => match nnn {
                0x0E0 => Command::Clear(i),
                0x0EE => Command::SubroutinePop(i),
                _ => Command::Nop(i),
            },
            0x1 => Command::Jump(i, nnn),
            0x2 => Command::SubroutinePush(i, nnn),
            0x3 => Command::SkipEqI(i, x, nn),
            0x4 => Command::SkipNeqI(i, x, nn),
            0x5 => Command::SkipEq(i, x, y),
            0x6 => Command::Set(i, x, nn),
            0x7 => Command::Add(i, x, nn),
            0x8 => match n {
                0x0 => Command::Arithmetic(i, Operation::Set(x, y)),
                0x1 => Command::Arithmetic(i, Operation::Or(x, y)),
                0x2 => Command::Arithmetic(i, Operation::And(x, y)),
                0x3 => Command::Arithmetic(i, Operation::Xor(x, y)),
                0x4 => Command::Arithmetic(i, Operation::Add(x, y)),
                0x5 => Command::Arithmetic(i, Operation::Sub(x, y)),
                0x7 => Command::Arithmetic(i, Operation::Sub(y, x)),
                0x6 => Command::Arithmetic(i, Operation::ShiftR(x, y)),
                0xE => Command::Arithmetic(i, Operation::ShiftL(x, y)),
                _ => Command::Nop(i),
            },
            0x9 => Command::SkipNeq(i, x, y),
            0xA => Command::SetIdx(i, nnn),
            0xB => Command::JumpX(i, nnn),
            0xC => Command::Random(i, x, nn),
            0xD => Command::Display {
                instruction: i,
                x,
                y,
                height: n,
            },
            0xE => match nn {
                0x9E => Command::SkipOnKey(i, x),
                0xA1 => Command::SkipOffKey(i, x),
                _ => Command::Nop(i),
            },
            0xF => match nn {
                0x07 => Command::ReadDelay(i, x),
                0x15 => Command::SetDelay(i, x),
                0x18 => Command::SetSound(i, x),
                0x1E => Command::AddIdx(i, x),
                0x0A => Command::GetKey(i, x),
                0x29 => Command::FontChar(i, x),
                0x33 => Command::BCD(i, x),
                0x55 => Command::StoreMem(i, x),
                0x65 => Command::ReadMem(i, x),
                _ => Command::Nop(i),
            },
            _ => Command::Nop(i),
        }
    }
}

#[cfg(test)]
mod test_command {
    use crate::{Command, Operation};

    #[test]
    fn decode() {
        let tests = vec![
            (0x00e0, Command::Clear(0x00e0)),
            (0x1abc, Command::Jump(0x1abc, 0xabc)),
            (0x2abc, Command::SubroutinePush(0x2abc, 0xabc)),
            (0x00ee, Command::SubroutinePop(0x00ee)),
            (0x3abc, Command::SkipEqI(0x3abc, 0xa, 0xbc)),
            (0x4abc, Command::SkipNeqI(0x4abc, 0xa, 0xbc)),
            (0x5ab0, Command::SkipEq(0x5ab0, 0xa, 0xb)),
            (0x9ab0, Command::SkipNeq(0x9ab0, 0xa, 0xb)),
            (0x6abc, Command::Set(0x6abc, 0xa, 0xbc)),
            (0x7abc, Command::Add(0x7abc, 0xa, 0xbc)),
            (0x8ab0, Command::Arithmetic(0x8ab0, Operation::Set(0xa, 0xb))),
            (0x8ab1, Command::Arithmetic(0x8ab1, Operation::Or(0xa, 0xb))),
            (0x8ab2, Command::Arithmetic(0x8ab2, Operation::And(0xa, 0xb))),
            (0x8ab3, Command::Arithmetic(0x8ab3, Operation::Xor(0xA, 0xB))),
            (0x8ab4, Command::Arithmetic(0x8ab4, Operation::Add(0xA, 0xB))),
            (0x8ab5, Command::Arithmetic(0x8ab5, Operation::Sub(0xA, 0xB))),
            (0x8ab7, Command::Arithmetic(0x8ab7, Operation::Sub(0xB, 0xA))),
            (0x8ab6, Command::Arithmetic(0x8ab6, Operation::ShiftR(0xA, 0xB))),
            (0x8abe, Command::Arithmetic(0x8abe, Operation::ShiftL(0xA, 0xB))),
            (0xaabc, Command::SetIdx(0xaabc, 0xABC)),
            (0xbabc, Command::JumpX(0xbabc, 0xABC)),
            (0xcabc, Command::Random(0xcabc, 0xA, 0xBC)),
            (
                0xdabc,
                Command::Display {
                    instruction: 0xdabc,
                    x: 0xA,
                    y: 0xB,
                    height: 0xC,
                },
            ),
            (0xea9e, Command::SkipOnKey(0xea9e, 0xA)),
            (0xeaa1, Command::SkipOffKey(0xeaa1, 0xA)),
            (0xfa07, Command::ReadDelay(0xfa07, 0xA)),
            (0xfa15, Command::SetDelay(0xfa15, 0xA)),
            (0xfa18, Command::SetSound(0xfa18, 0xA)),
            (0xfa1e, Command::AddIdx(0xfa1e, 0xA)),
            (0xfa0a, Command::GetKey(0xfa0a, 0xA)),
            (0xfa29, Command::FontChar(0xfa29, 0xA)),
            (0xfa33, Command::BCD(0xfa33, 0xA)),
            (0xfa55, Command::StoreMem(0xfa55, 0xA)),
            (0xfa65, Command::ReadMem(0xfa65, 0xA)),
        ];
        for test in tests {
            assert_eq!(Command::from(test.0), test.1);
        }
    }
}

trait Memory {
    fn read(&self, address: usize, n_bytes: usize) -> Result<&[u8], ErrMemory>;
    fn write(&mut self, address: usize, value: &[u8]) -> Result<(), ErrMemory>;
}

#[derive(Debug, PartialEq)]
enum ErrMemory {
    Overflow,
}

impl Memory for [u8] {
    fn read(&self, address: usize, len: usize) -> Result<&[u8], ErrMemory> {
        if len == 0 {
            Ok(&[])
        } else if address + len > self.len() {
            Err(ErrMemory::Overflow)
        } else {
            Ok(&self[address..address + len])
        }
    }

    fn write(&mut self, address: usize, value: &[u8]) -> Result<(), ErrMemory> {
        let n = value.len();
        if n == 0 {
            Ok(())
        } else if address + n > self.len() {
            Err(ErrMemory::Overflow)
        } else {
            self[address..address + n].copy_from_slice(value);
            Ok(())
        }
    }
}

const MEM_SIZE: usize = 0x1000;

const FONT_DATA: [u8; 80] = [
    0xf0, 0x90, 0x90, 0x90, 0xf0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xf0, 0x10, 0xf0, 0x80, 0xf0, // 2
    0xf0, 0x10, 0xf0, 0x10, 0xf0, // 3
    0x90, 0x90, 0xf0, 0x10, 0x10, // 4
    0xf0, 0x80, 0xf0, 0x10, 0xf0, // 5
    0xf0, 0x80, 0xf0, 0x90, 0xf0, // 6
    0xf0, 0x10, 0x20, 0x40, 0x40, // 7
    0xf0, 0x90, 0xf0, 0x90, 0xf0, // 8
    0xf0, 0x90, 0xf0, 0x10, 0xf0, // 9
    0xf0, 0x90, 0xf0, 0x90, 0x90, // A
    0xe0, 0x90, 0xe0, 0x90, 0xe0, // B
    0xf0, 0x80, 0x80, 0x80, 0xf0, // C
    0xe0, 0x90, 0x90, 0x90, 0xe0, // D
    0xf0, 0x80, 0xf0, 0x80, 0xf0, // E
    0xf0, 0x80, 0xf0, 0x80, 0x80, // F
];

#[cfg(test)]
mod test_array_mem {
    use crate::{ErrMemory, Memory};

    #[test]
    fn can_read_and_write() {
        let mut m: [u8; 0x1000] = [0; 0x1000];
        let tests = vec![
            (0, vec![0, 1, 2]),
            (0, vec![1, 2, 3]),
            (8, vec![0, 1]),
            (8, vec![1]),
            (0, vec![]),
        ];
        for (idx, xs) in tests {
            m.write(idx, &xs).unwrap();
            assert_eq!(m.read(idx, xs.len()).unwrap(), xs);
        }
    }

    #[test]
    fn set_invalid_index() {
        let mut m: [u8; 0x1000] = [0; 0x1000];
        assert!(m.write(0x1000, &[]).is_ok());
        assert!(m.write(0x1001, &[]).is_ok());
        assert_eq!(m.write(0x1001, &[0]).unwrap_err(), ErrMemory::Overflow);
    }

    #[test]
    fn get_invalid_index() {
        let m: [u8; 0x1000] = [0; 0x1000];
        assert!(m.read(0x1000, 0).is_ok());
        assert!(m.read(0x1001, 0).is_ok());
        assert_eq!(m.read(0x1001, 1).unwrap_err(), ErrMemory::Overflow);
    }
}

struct Timer {
    val: Arc<Mutex<u8>>,
    ticker: JoinHandle<()>,
}

impl Timer {
    fn new(init: u8, period: time::Duration) -> Self {
        let val = Arc::new(Mutex::new(init));
        let val_c = Arc::clone(&val);
        let ticker = thread::spawn(move || loop {
            {
                let mut val = val_c.lock().unwrap();
                match *val {
                    0 => (),
                    x => *val = x - 1,
                }
            }
            thread::sleep(period);
        });
        Self { val, ticker }
    }

    fn set(&self, val: u8) {
        *self.val.lock().unwrap() = val;
    }

    fn get(&self) -> u8 {
        *self.val.lock().unwrap()
    }
}

/// Period corresponding to 60 Hz.
const TIMER_INTERVAL: time::Duration = time::Duration::from_nanos(16_666_666);

/// Period corresponding to 600 Hz.
const EMULATOR_INTERVAL: time::Duration = time::Duration::from_nanos(1_666_666);

#[cfg(test)]
mod test_timer {
    use std::{thread, time};

    use crate::Timer;

    #[test]
    fn set_and_get() {
        let timer = Timer::new(0, time::Duration::MAX);
        timer.set(8);
        assert_eq!(timer.get(), 8);
    }

    #[test]
    fn decrements() {
        let timer = Timer::new(8, time::Duration::from_nanos(1));
        thread::sleep(time::Duration::from_millis(1));
        assert_eq!(timer.get(), 0);
    }
}

type Pixels = Arc<Mutex<[bool; 64 * 32]>>;

struct Screen {
    pixels: Pixels,
    updates: mpsc::Sender<Update>,
}

impl Screen {
    fn new(tx: mpsc::Sender<Update>) -> Self {
        Self {
            updates: tx,
            pixels: Arc::new(Mutex::new([false; 64 * 32])),
        }
    }

    fn set(&mut self, i: usize, j: usize) {
        if i >= 32 || j >= 64 {
            return;
        }
        let mut pixels = self.pixels.lock().unwrap();
        pixels[i * 64 + j] = true;
    }

    /// Flips the pixel at coordinate given and returns the old value, if any. If the coordinate is
    /// off the screen then this is a noop.
    fn flip(&self, i: usize, j: usize, pixels: &mut [bool; 64 * 32]) -> Option<bool> {
        if i >= 32 || j >= 64 {
            return None;
        }
        let old = pixels[i * 64 + j];
        pixels[i * 64 + j] = !old;
        Some(old)
    }

    /// Draws a sprite at the coordinate given. Only the starting coordinate wraps around, and all
    /// other pixels are not drawn if they fall off the screen. Returns true if any pixel was
    /// turned off.
    fn draw_sprite(&mut self, i: usize, j: usize, sprite: &[u8]) -> bool {
        let mut i = i % 32;
        let j = j % 64;
        let mut flag = false;

        let mut pixels = self.pixels.lock().unwrap();
        for byte in sprite {
            let bits = split_u8_into_bits(*byte);
            for d in 0..8 {
                if bits[d] == false {
                    continue;
                }
                match self.flip(i, j + d, &mut pixels) {
                    Some(old) if old == false => flag = true,
                    _ => (),
                }
            }
            i += 1;
        }

        flag
    }

    fn clear(&mut self) {
        let mut pixels = self.pixels.lock().unwrap();
        for i in 0..pixels.len() {
            pixels[i] = false;
        }
    }
}

#[cfg(test)]
mod test_pixels {
    use crate::Screen;
    use rand::prelude::*;
    use std::sync::mpsc;

    #[test]
    fn can_set_out_of_screen() {
        let (tx, _) = mpsc::channel();
        let mut pxs = Screen::new(tx);
        pxs.set(33, 65);
    }

    #[test]
    fn clear() {
        let (tx, _) = mpsc::channel();
        let mut pxs = Screen::new(tx);
        let mut rng = thread_rng();
        for _ in 0..64 {
            let i = rng.gen_range(0..32);
            let j = rng.gen_range(0..64);
            pxs.set(i, j);
        }
        assert_ne!(*pxs.pixels.lock().unwrap(), [false; 64 * 32]);

        pxs.clear();
        assert_eq!(*pxs.pixels.lock().unwrap(), [false; 64 * 32]);
    }
}

enum Update {
    Nop,
    Exit,
    ProgramCounter(usize),
}

struct UI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    pixels: Pixels,
    updates: mpsc::Receiver<Update>,

    /// All instructions of the program.
    instructions: Vec<Command>,
    current_instruction: usize,
}

impl UI {
    fn new(rx: mpsc::Receiver<Update>, pixels: Pixels, instructions: Vec<Command>) -> Self {
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).unwrap();

        Self {
            terminal,
            updates: rx,
            pixels,
            instructions,
            current_instruction: 0,
        }
    }

    fn run(&mut self) {
        terminal::enable_raw_mode().unwrap();
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();

        loop {
            match self.updates.try_recv() {
                Ok(Update::Exit) | Err(TryRecvError::Disconnected) => break,
                Ok(Update::ProgramCounter(pc)) => self.current_instruction = (pc - 0x200) / 2,
                _ => (),
            }
            self.terminal
                .draw(|f| {
                    ui(f, self.pixels.clone(), &self.instructions, self.current_instruction);
                })
                .unwrap();
        }

        terminal::disable_raw_mode().unwrap();
        crossterm::execute!(self.terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture).unwrap();
        self.terminal.show_cursor().unwrap();
    }
}

fn ui<B: Backend>(
    f: &mut Frame<B>,
    pixels: Arc<Mutex<[bool; 64 * 32]>>,
    instructions: &[Command],
    current_instruction: usize,
) {
    // Split screen into top and bottom halves.
    let screen = Layout::default()
        .direction(layout::Direction::Vertical)
        .constraints([layout::Constraint::Percentage(50), layout::Constraint::Percentage(50)].as_ref())
        .split(f.size());

    let top = screen[0];
    let bot = screen[1];

    // Split each half into a left and right portion.
    let top = Layout::default()
        .direction(layout::Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(top);
    let bot = Layout::default()
        .direction(layout::Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(bot);

    // Create the game screen.
    let main_screen = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title("CHIP-8"))
        .marker(Marker::Block)
        .x_bounds([0.0, 64.0])
        .y_bounds([0.0, 32.0])
        .paint(|ctx| {
            let mut coords: Vec<(f64, f64)> = vec![];
            {
                let pixels = pixels.lock().unwrap();
                coords = pixels.iter().enumerate().fold(coords, |mut accum, px| {
                    if *px.1 {
                        let x = px.0 % 64;
                        let y = 32 - px.0 / 64; // tui-rs takes (0, 0) to be bottom-left
                        accum.push((x as f64, y as f64));
                    }
                    accum
                });
            }
            ctx.draw(&Points {
                coords: &coords[..],
                color: Color::White,
            })
        });

    let instructions_list: Vec<ListItem> = instructions
        .iter()
        .map(|i| ListItem::new(Spans::from(vec![Span::raw(i.to_string())])))
        .collect();
    let mut instructions_state = ListState::default();
    instructions_state.select(Some(current_instruction));
    let instructions_widget = List::new(instructions_list)
        .block(Block::default().borders(Borders::ALL).title("Instructions"))
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Black)
                .bg(Color::White),
        )
        .highlight_symbol(">");

    f.render_widget(main_screen, top[0]);
    f.render_stateful_widget(instructions_widget, bot[0], &mut instructions_state);
}
