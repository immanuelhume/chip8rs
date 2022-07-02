use crossterm::{
    self,
    event::{DisableMouseCapture, EnableMouseCapture, KeyCode},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    env, error,
    io::{self, Read},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time,
};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{self, Constraint, Layout},
    style::Color,
    symbols::Marker,
    widgets::{
        canvas::{Canvas, Points},
        Block, Borders,
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
    let mut ui = UI::new(rx);

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
    display: Display,
    idx_reg: usize,
    registers: [u8; 0x10],
    stack: Vec<u16>,
    delay_timer: Timer,
    sound_timer: Timer,
}

impl Emulator {
    fn new(tx: mpsc::Sender<Update>, poll_timeout: time::Duration) -> Self {
        let mut mem = [0; 0x1000];
        mem.write(0x100, &FONT_DATA).unwrap(); // Start at 0x200.
        Self {
            pc: 0x200,
            mem,
            poll_timeout,
            updates: tx.clone(),
            display: Display::new(tx),
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
            let next = self
                .fetch_next_and_decode()
                .expect("could not decode next instruction");
            self.exec(next);

            if crossterm::event::poll(self.poll_timeout)? {
                if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                    match key.code {
                        KeyCode::Char('q') => {
                            self.updates.send(Update::Exit)?;
                            break;
                        }
                        _ => (),
                    }
                }
            }
        }

        Ok(())
    }

    fn exec(&mut self, cmd: Command) {
        match cmd {
            Command::Clear => self.display.clear(),
            Command::Jump(addr) => self.pc = addr,
            Command::Set(reg, val) => self.registers[reg] = val,
            Command::Add(reg, val) => self.registers[reg] += val,
            Command::SetIdx(v) => self.idx_reg = v,
            Command::Display { x, y, height: n } => {
                let i = self.registers[y] as usize;
                let j = self.registers[x] as usize;
                let sprite = self.mem.read(self.idx_reg, n).unwrap();

                let is_px_turned_off = self.display.draw_sprite(i, j, sprite);
                self.registers[0xF] = if is_px_turned_off { 1 } else { 0 };
            }
            _ => (),
        }
    }
}

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
    let tests = vec![
        (0xABCD, 0xAB, 0xCD),
        (0x0000, 0x00, 0x00),
        (0x0A0A, 0x0A, 0x0A),
    ];
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
    let tests = vec![(
        0b01010101,
        [false, true, false, true, false, true, false, true],
    )];
    for test in tests {
        assert_eq!(split_u8_into_bits(test.0), test.1);
    }
}

#[derive(Debug, PartialEq)]
enum Command {
    Nop,
    Clear,
    Jump(usize),
    Set(usize, u8),
    Add(usize, u8),
    SetIdx(usize),
    Display { x: usize, y: usize, height: usize },
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
            0 => Command::Clear,
            1 => Command::Jump(nnn),
            6 => Command::Set(x, nn),
            7 => Command::Add(x, nn),
            0xA => Command::SetIdx(nnn),
            0xD => Command::Display { x, y, height: n },
            _ => Command::Nop,
        }
    }
}

#[cfg(test)]
mod test_command {
    use crate::Command;

    #[test]
    fn create_from_instruction() {
        let tests = vec![
            (0x00E0, Command::Clear),
            (0x1ABC, Command::Jump(0xABC)),
            (0x6ABC, Command::Set(0xA, 0xBC)),
            (0x7ABC, Command::Add(0xA, 0xBC)),
            (0xAABC, Command::SetIdx(0xABC)),
            (
                0xDABC,
                Command::Display {
                    x: 0xA,
                    y: 0xB,
                    height: 0xC,
                },
            ),
            (0x2222, Command::Nop),
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
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
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

struct Display {
    pixels: Pixels,
    updates: mpsc::Sender<Update>,
}

impl Display {
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

        {
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
        }

        self.updates
            .send(Update::Display(self.pixels.clone()))
            .unwrap();
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
    use crate::Display;
    use rand::prelude::*;
    use std::sync::mpsc;

    #[test]
    fn can_set_out_of_screen() {
        let (tx, _) = mpsc::channel();
        let mut pxs = Display::new(tx);
        pxs.set(33, 65);
    }

    #[test]
    fn clear() {
        let (tx, _) = mpsc::channel();
        let mut pxs = Display::new(tx);
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
    Log(String),
    Display(Pixels),
}

struct UI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    updates: mpsc::Receiver<Update>,
}

impl UI {
    fn new(rx: mpsc::Receiver<Update>) -> Self {
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).unwrap();

        Self {
            terminal,
            updates: rx,
        }
    }

    fn run(&mut self) {
        terminal::enable_raw_mode().unwrap();
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();

        loop {
            let update = self.updates.recv();
            if update.is_err() {
                break;
            }
            match update.unwrap_or(Update::Nop) {
                Update::Display(pxs) => {
                    self.terminal
                        .draw(|f| {
                            ui(f, pxs);
                        })
                        .unwrap();
                }
                Update::Exit => break,
                _ => (),
            }
        }

        terminal::disable_raw_mode().unwrap();
        crossterm::execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .unwrap();
        self.terminal.show_cursor().unwrap();
    }
}

fn ui<B: Backend>(f: &mut Frame<B>, pixels: Arc<Mutex<[bool; 64 * 32]>>) {
    let screen = Layout::default()
        .direction(layout::Direction::Vertical)
        .constraints(
            [
                layout::Constraint::Percentage(50),
                layout::Constraint::Percentage(50),
            ]
            .as_ref(),
        )
        .split(f.size());

    let top = screen[0];
    let bot = screen[1];

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

    let top = Layout::default()
        .direction(layout::Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(top);

    f.render_widget(main_screen, top[0]);
}
