use crate::instruction::Instruction;
use crate::AppUpdate;
use crate::EmuState;
use crate::Pixels;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use std::io;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};
use tui::backend::{Backend, CrosstermBackend};
use tui::layout::{Constraint, Direction, Layout};
use tui::style::{Color, Modifier, Style};
use tui::text::{Span, Spans};
use tui::widgets::canvas::{Canvas, Points};
use tui::widgets::{Block, Borders, List, ListItem, ListState};
use tui::Frame;
use tui::Terminal;

pub struct App {
    pixels: Arc<Mutex<Pixels>>,
    instructions: Vec<Instruction>,
    current_instruction: u16,
    registers: Arc<Mutex<[u8; 0x10]>>,
    stack: Arc<Mutex<Vec<u16>>>,
    updates_rx: Receiver<AppUpdate>,
    emu_state: EmuState,
}

impl App {
    pub fn new(
        pixels: Arc<Mutex<Pixels>>,
        registers: Arc<Mutex<[u8; 0x10]>>,
        stack: Arc<Mutex<Vec<u16>>>,
        instructions: Vec<Instruction>,
        updates_rx: Receiver<AppUpdate>,
    ) -> Self {
        Self {
            pixels,
            instructions,
            current_instruction: 0x200,
            registers,
            stack,
            updates_rx,
            emu_state: EmuState::Normal,
        }
    }

    pub fn spin(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

        loop {
            match self.updates_rx.try_recv() {
                Ok(update) => match update {
                    AppUpdate::Exit => break,
                    AppUpdate::PC(pc) => self.current_instruction = (pc - 0x200) / 2,
                    AppUpdate::State(state) => self.emu_state = state,
                },
                Err(err) => match err {
                    TryRecvError::Empty => (),
                    TryRecvError::Disconnected => break,
                },
            }
            terminal.draw(|f| self.ui(f))?;
        }

        terminal::disable_raw_mode()?;
        crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn ui<B: Backend>(&self, f: &mut Frame<B>) {
        // Split screen into top and bottom halves.
        let screen = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(f.size());

        let top = screen[0];
        let bot = screen[1];

        // Split each half into a left, middle, and right portion.
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(25),
                    Constraint::Percentage(50),
                    Constraint::Percentage(25),
                ]
                .as_ref(),
            )
            .split(top);
        let bot = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(25),
                    Constraint::Percentage(50),
                    Constraint::Percentage(25),
                ]
                .as_ref(),
            )
            .split(bot);

        // Create the game screen.
        {
            let widget = Canvas::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("CHIP-8 ({})", self.emu_state)),
                )
                // .marker(Marker::Dot)
                .x_bounds([0.0, 64.0])
                .y_bounds([0.0, 32.0])
                .paint(|ctx| {
                    let mut coords: Vec<(f64, f64)> = vec![];
                    {
                        let pixels = self.pixels.lock().unwrap();
                        coords = pixels.0.iter().enumerate().fold(coords, |mut accum, px| {
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
            f.render_widget(widget, top[1]);
        }

        // Create a list of instructions.
        {
            let instructions: Vec<ListItem> = self
                .instructions
                .iter()
                .enumerate()
                .map(|i| {
                    ListItem::new(Spans::from(vec![
                        Span::raw(format!("{:#05x}", i.0 * 2 + 0x200)),
                        Span::raw(" "),
                        Span::raw(i.1.to_string()),
                    ]))
                })
                .collect();

            let mut state = ListState::default();
            state.select(Some(self.current_instruction as usize));

            let widget = List::new(instructions)
                .block(Block::default().borders(Borders::ALL).title("Instructions"))
                .highlight_style(
                    Style::default()
                        .add_modifier(Modifier::BOLD)
                        .fg(Color::Black)
                        .bg(Color::White),
                )
                .highlight_symbol(">");
            f.render_stateful_widget(widget, bot[1], &mut state);
        }

        // Print out all the registers.
        {
            let registers: Vec<ListItem> = self
                .registers
                .lock()
                .unwrap()
                .iter()
                .enumerate()
                .map(|x| {
                    ListItem::new(Spans::from(vec![
                        Span::styled(format!("V{:<3X}", x.0), Style::default().fg(Color::Blue)),
                        Span::styled(format!("{:#010b}", x.1), Style::default().add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::raw(format!("({:#04x})", x.1)),
                    ]))
                })
                .collect();
            let widget = List::new(registers).block(Block::default().borders(Borders::ALL).title("Registers"));
            f.render_widget(widget, bot[0]);
        }

        // Print out key bindings.
        {
            let key_bindings: Vec<ListItem> = KEY_BINDINGS
                .iter()
                .map(|(key, text)| {
                    ListItem::new(Spans::from(vec![
                        Span::styled(
                            format!("{:<8}", *key),
                            Style::default().add_modifier(Modifier::BOLD).fg(Color::Blue),
                        ),
                        Span::raw(" "),
                        Span::raw(*text),
                    ]))
                })
                .collect();
            let widget = List::new(key_bindings).block(Block::default().borders(Borders::ALL).title("Key bindings"));
            f.render_widget(widget, bot[2]);
        }
    }
}

const KEY_BINDINGS: [(&str, &str); 4] = [
    ("<i>", "Enter debug mode"),
    ("<o>", "Exit emulator"),
    ("<p>", "Pause emulation"),
    ("<Enter>", "(In debug mode) Step next"),
];
