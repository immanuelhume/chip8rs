use crate::ui::App;
use crossterm::event::{self, Event, KeyCode};
use instruction::{Instruction, Operation};
use std::io::{self, Read};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::{env, thread};
use std::{fmt, time};
use tui::backend::CrosstermBackend;
use tui::Terminal;

mod instruction;
mod state;
mod timer;
mod ui;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the entire CHIP-8 file into memory.
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];
    let mut rom = vec![];
    {
        let mut file = std::fs::File::open(filename)?;
        file.read_to_end(&mut rom)?;
    }

    let pixels = Arc::new(Mutex::new(Pixels::new()));
    let registers = Arc::new(Mutex::new([0; 16]));
    let stack = Arc::new(Mutex::new(vec![0; 16]));
    let (tx, rx) = mpsc::sync_channel(2);
    let mut e = Emulator::new(pixels.clone(), registers.clone(), stack.clone(), &rom, tx);

    let instructions: Vec<Instruction> = rom
        .chunks(2)
        .map(|xs| {
            if let [a, b] = *xs {
                let c = (a as u16) << 8 | b as u16;
                Instruction::from(c)
            } else {
                // panic!("wrong number of bytes read from rom")
                Instruction::Nop(0000)
            }
        })
        .collect();
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut app = App::new(pixels, registers, stack, instructions, rx);

    // We copied the rom into the emulator and the UI (as instructions) so we can drop it here
    // before starting stuff.
    drop(rom);

    thread::spawn(move || {
        while let Some(state_fn) = e.state_fn.0(&mut e) {
            e.state_fn = state_fn;
            thread::sleep(EMULATOR_PERIOD);
        }
    });
    app.spin(&mut terminal).unwrap();

    Ok(())
}

/// Period corresponding to 60 Hz.
const TIMER_PERIOD: time::Duration = time::Duration::from_nanos(16_666_666);
/// Period corresponding to 600 Hz.
const EMULATOR_PERIOD: time::Duration = time::Duration::from_nanos(1_666_666);
const POLL_TIMEOUT: time::Duration = time::Duration::from_millis(1);
/// Address in memory where the fonts start.
const FONT_IDX: u16 = 0x100;
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
/// Maps a hexadecimal digit (index in the array) to the corresponding keycode.
const KEY_MAP: [char; 16] = [
    'x', '1', '2', '3', 'q', 'w', 'e', 'a', 's', 'd', 'z', 'c', '4', 'r', 'f', 'v',
];

pub struct Emulator {
    pc: u16,
    /// Memory, words must be stored in big endian.
    mem: [u8; 0x1000],
    /// Representation of pixels on the screen.
    pixels: Arc<Mutex<Pixels>>,
    /// The I register.
    i: u16,
    registers: Arc<Mutex<[u8; 0x10]>>,
    stack: Arc<Mutex<Vec<u16>>>,
    delay_timer: timer::Timer,
    sound_timer: timer::Timer,

    updates_tx: SyncSender<AppUpdate>,
    state_fn: state::StateFn,
}

impl Emulator {
    fn new(
        pixels: Arc<Mutex<Pixels>>,
        registers: Arc<Mutex<[u8; 0x10]>>,
        stack: Arc<Mutex<Vec<u16>>>,
        rom: &[u8],
        updates_tx: SyncSender<AppUpdate>,
    ) -> Self {
        let mut mem = [0; 0x1000];
        mem.write(FONT_IDX, &FONT_DATA);
        mem.write(0x200, rom);

        Self {
            pc: 0x200,
            mem,
            pixels,
            i: 0,
            registers,
            stack,
            delay_timer: timer::Timer::new(0, TIMER_PERIOD),
            sound_timer: timer::Timer::new(0, TIMER_PERIOD),
            updates_tx,
            state_fn: state::StateFn(state::normal),
        }
    }

    /// Fetches the next instruction pointed to by the program counter and
    /// decodes it.
    fn fetch_next_and_decode(&mut self) -> Result<Instruction, Box<dyn std::error::Error + '_>> {
        self.updates_tx.send(AppUpdate::PC(self.pc))?;
        if let [a, b] = *self.mem.read(self.pc, 2) {
            self.pc += 2;
            let i: u16 = (0 | a as u16) << 8 | b as u16;
            Ok(Instruction::from(i))
        } else {
            // This should never happen unless the ROM or the PC is messed up.
            panic!()
        }
    }
}

/// Some type of update which the UI must know about. Note that for the most
/// part, the UI shares state with the emulator via mutexes. But we can also use
/// these updates if it is more ergonomic.
#[derive(Debug)]
pub enum AppUpdate {
    Exit,
    PC(u16),
}

impl std::error::Error for AppUpdate {}
impl fmt::Display for AppUpdate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppUpdate::Exit => write!(f, "exit emulator"),
            AppUpdate::PC(pc) => write!(f, "new pc: {}", pc),
        }
    }
}

pub struct Pixels([bool; 64 * 32]);

impl Pixels {
    fn new() -> Self {
        Self([false; 64 * 32])
    }

    fn set(&mut self, i: usize, j: usize) {
        if i >= 32 || j >= 64 {
            return;
        }
        self.0[i * 64 + j] = true;
    }

    /// Flips the pixel at coordinate given and returns the old value, if any.
    /// If the coordinate is off the screen then this is a noop.
    fn flip(&mut self, i: usize, j: usize) -> Option<bool> {
        if i >= 32 || j >= 64 {
            return None;
        }
        let old = self.0[i * 64 + j];
        self.0[i * 64 + j] = !old;
        Some(old)
    }

    /// Draws a sprite at the coordinate given. Only the starting coordinate
    /// wraps around, and all other pixels are not drawn if they fall off the
    /// screen. Returns true if any pixel was turned off.
    fn draw_sprite(&mut self, i: usize, j: usize, sprite: &[u8]) -> bool {
        let mut i = i % 32;
        let j = j % 64;
        let mut flag = false;

        for byte in sprite {
            let bits = split_u8_into_bits(*byte);
            for d in 0..8 {
                if bits[d] == false {
                    continue;
                }
                match self.flip(i, j + d) {
                    Some(old) if old == false => flag = true,
                    _ => (),
                }
            }
            i += 1;
        }

        flag
    }

    /// Clears the screen.
    fn clear(&mut self) {
        for i in 0..self.0.len() {
            self.0[i] = false;
        }
    }
}

#[cfg(test)]
mod test_pixels {
    use crate::Pixels;
    use rand::prelude::*;

    #[test]
    fn can_set_out_of_screen() {
        let mut pxs = Pixels::new();
        pxs.set(33, 65);
    }

    #[test]
    fn clear() {
        let mut pxs = Pixels::new();
        let mut rng = thread_rng();
        for _ in 0..64 {
            let i = rng.gen_range(0..32);
            let j = rng.gen_range(0..64);
            pxs.set(i, j);
        }
        assert_ne!(pxs.0, [false; 64 * 32]);

        pxs.clear();
        assert_eq!(pxs.0, [false; 64 * 32]);
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
    let tests = vec![(
        0b01010101,
        [false, true, false, true, false, true, false, true],
        (0b00000000, [false; 8]),
        (0b11111111, [true; 8]),
    )];
    for test in tests {
        assert_eq!(split_u8_into_bits(test.0), test.1);
    }
}

trait Memory {
    fn read(&self, address: u16, n_bytes: u16) -> &[u8];
    fn write(&mut self, address: u16, value: &[u8]);
}

impl Memory for [u8] {
    fn read(&self, address: u16, n_bytes: u16) -> &[u8] {
        let address = address as usize;
        let n_bytes = n_bytes as usize;
        if n_bytes == 0 {
            &[]
        } else if address + n_bytes > self.len() {
            panic!(
                "tried to read {} byte(s) from address {:#04x} but the memory only has {:#04x} bytes",
                n_bytes,
                address,
                self.len()
            );
        } else {
            &self[address..address + n_bytes]
        }
    }

    fn write(&mut self, address: u16, value: &[u8]) {
        let address = address as usize;
        let n = value.len();
        if n == 0 {
            return;
        } else if address + n > self.len() {
            panic!(
                "tried to write {} byte(s) to address {:#04x} but the memory only has {:#04x} bytes",
                n,
                address,
                self.len()
            );
        } else {
            self[address..address + n].copy_from_slice(value);
        }
    }
}

#[cfg(test)]
mod test_array_mem {
    use crate::Memory;

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
            m.write(idx, &xs);
            assert_eq!(m.read(idx, xs.len() as u16), xs);
        }
    }

    #[test]
    fn set_invalid_index_empty() {
        let mut m: [u8; 0x1000] = [0; 0x1000];
        m.write(0x1000, &[]);
        m.write(0x1001, &[]);
    }

    #[test]
    #[should_panic(expected = "tried to write 1 byte(s) to address 0x1001 but the memory only has 0x1000 bytes")]
    fn set_invalid_index_nonempty() {
        let mut m: [u8; 0x1000] = [0; 0x1000];
        m.write(0x1001, &[0]);
    }

    #[test]
    fn get_invalid_index_empty() {
        let m: [u8; 0x1000] = [0; 0x1000];
        m.read(0x1000, 0);
        m.read(0x1001, 0);
    }

    #[test]
    #[should_panic(expected = "tried to read 1 byte(s) from address 0x1001 but the memory only has 0x1000 bytes")]
    fn get_invalid_index_nonempty() {
        let m: [u8; 0x1000] = [0; 0x1000];
        m.read(0x1001, 1);
    }
}

/// Executes a single instruction.
pub fn exec(i: Instruction, e: &mut Emulator) -> Result<(), Box<dyn std::error::Error + '_>> {
    use Instruction::*;
    match i {
        Clear(_) => e.pixels.lock()?.clear(),
        Jump(_, addr) => e.pc = addr,
        SubroutinePush(_, addr) => {
            let mut stack = e.stack.lock()?;
            stack.push(e.pc);
            e.pc = addr;
        }
        SubroutinePop(_) => {
            let mut stack = e.stack.lock()?;
            e.pc = stack.pop().unwrap();
        }
        SkipEqI(_, reg, val) => {
            if e.registers.lock()?[reg as usize] == val {
                e.pc += 2;
            }
        }
        SkipNeqI(_, reg, val) => {
            if e.registers.lock()?[reg as usize] != val {
                e.pc += 2;
            }
        }
        SkipEq(_, reg1, reg2) => {
            let registers = e.registers.lock()?;
            if registers[reg1 as usize] == registers[reg2 as usize] {
                e.pc += 2;
            }
        }
        SkipNeq(_, reg1, reg2) => {
            let registers = e.registers.lock()?;
            if registers[reg1 as usize] != registers[reg2 as usize] {
                e.pc += 2;
            }
        }
        Arithmetic(_, op) => {
            let mut registers = e.registers.lock()?;
            match op {
                Operation::Set(x, y) => registers[x as usize] = registers[y as usize],
                Operation::Or(x, y) => registers[x as usize] |= registers[y as usize],
                Operation::And(x, y) => registers[x as usize] &= registers[y as usize],
                Operation::Xor(x, y) => registers[x as usize] ^= registers[y as usize],
                Operation::Add(x, y) => {
                    let (res, overflow) = registers[x as usize].overflowing_add(registers[y as usize]);
                    registers[x as usize] = res;
                    if overflow {
                        registers[0xF] = 1;
                    }
                }
                Operation::Sub(x, y) => {
                    let (res, overflow) = registers[x as usize].overflowing_sub(registers[y as usize]);
                    registers[x as usize] = res;
                    registers[0xF] = if overflow { 0 } else { 1 };
                }
                Operation::ShiftR(x, y) => {
                    let a = registers[y as usize];
                    registers[0xF] = a & 1;
                    registers[x as usize] = a >> 1;
                }
                Operation::ShiftL(x, y) => {
                    let a = registers[y as usize];
                    registers[0xF] = a & 0x80;
                    registers[x as usize] = a << 1;
                }
            }
        }
        Set(_, reg, val) => e.registers.lock()?[reg as usize] = val,
        Add(_, reg, val) => {
            let mut registers = e.registers.lock()?;
            registers[reg as usize] = registers[reg as usize].wrapping_add(val);
        }
        SetIdx(_, v) => e.i = v,
        JumpX(_, addr) => e.pc = e.registers.lock()?[0] as u16 + addr,
        Random(_, reg, val) => e.registers.lock()?[reg as usize] = rand::random::<u8>() & val,
        Display { x, y, n, .. } => {
            let mut registers = e.registers.lock()?;
            let i = registers[y as usize];
            let j = registers[x as usize];
            let sprite = e.mem.read(e.i, n as u16);

            let is_px_turned_off = e.pixels.lock()?.draw_sprite(i as usize, j as usize, sprite);
            registers[0xF] = if is_px_turned_off { 1 } else { 0 };
        }
        SkipOnKey(_, x) => {
            if event::poll(POLL_TIMEOUT)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char(KEY_MAP[e.registers.lock()?[x as usize] as usize]) {
                        e.pc += 2;
                    }
                }
            }
        }
        SkipOffKey(_, x) => {
            if event::poll(POLL_TIMEOUT)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char(KEY_MAP[e.registers.lock()?[x as usize] as usize]) {
                        e.pc -= 2; // dumb trick
                    }
                }
            }
            e.pc += 2;
        }
        ReadDelay(_, x) => e.registers.lock()?[x as usize] = e.delay_timer.get(),
        SetDelay(_, x) => e.delay_timer.set(e.registers.lock()?[x as usize]),
        SetSound(_, x) => e.sound_timer.set(e.registers.lock()?[x as usize]),
        AddIdx(_, x) => {
            let a = e.i;
            let mut registers = e.registers.lock()?;
            let sum = a + registers[x as usize] as u16;
            if sum > 0x0FFF && a <= 0x0FFF {
                registers[0xF] = 1;
            }
            e.i = sum;
        }
        GetKey(_, x) => loop {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char(c) = key.code {
                    match KEY_MAP.iter().position(|&x| x == c) {
                        Some(i) => {
                            e.registers.lock()?[x as usize] = i as u8;
                            break;
                        }
                        None => {
                            if c == 'o' {
                                return Err(Box::new(AppUpdate::Exit));
                            }
                        }
                    }
                }
            }
        },
        FontChar(_, x) => e.i = FONT_IDX + (e.registers.lock()?[x as usize] & 0x0F) as u16,
        BCD(_, x) => {
            let mut a = e.registers.lock()?[x as usize];
            for i in (0..3).rev() {
                e.mem.write(e.i + i, &[a % 10]);
                a /= 10;
            }
        }
        StoreMem(_, x) => e.mem.write(e.i, &e.registers.lock()?[0..=x as usize]),
        ReadMem(_, x) => e.registers.lock()?[0..=x as usize].copy_from_slice(e.mem.read(e.i, x as u16 + 1)),
        Nop(_) => (),
    }

    Ok(())
}
