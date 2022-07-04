use crate::instruction::Instruction;
use crate::AppUpdate;
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
use tui::symbols::Marker;
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

        // Split each half into a left and right portion.
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(top);
        let bot = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(bot);
        let top_right = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(top[1]);

        // Create the game screen.
        {
            let widget = Canvas::default()
                .block(Block::default().borders(Borders::ALL).title("CHIP-8"))
                .marker(Marker::Block)
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
            f.render_widget(widget, top[0]);
        }

        // Create a list of instructions.
        {
            let instructions: Vec<ListItem> = self
                .instructions
                .iter()
                .map(|i| ListItem::new(Spans::from(vec![Span::raw(i.to_string())])))
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
            f.render_stateful_widget(widget, bot[0], &mut state);
        }
    }
}