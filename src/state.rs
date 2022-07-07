use crate::{exec, AppUpdate, EmuState, Emulator, POLL_TIMEOUT};
use crossterm::event::{self, Event, KeyCode};
use std::time;

pub struct StateFn(pub fn(&mut Emulator) -> Option<StateFn>);

pub fn normal(e: &mut Emulator) -> Option<StateFn> {
    e.updates_tx
        .send(AppUpdate::State(EmuState::Normal))
        .expect("could not update emulator state to normal");
    match try_get_key(POLL_TIMEOUT) {
        None => {}
        Some(key) => match key {
            KeyCode::Char('p') => return Some(StateFn(paused)),
            KeyCode::Char('o') => {
                e.updates_tx.send(AppUpdate::Exit).unwrap();
                return None;
            }
            KeyCode::Char('i') => return Some(StateFn(debug)),
            _ => {}
        },
    }
    let i = e.fetch_next_and_decode().expect("could not decode next instruction");
    exec(i, e).expect("failed while executing instruction");
    Some(StateFn(normal))
}

fn paused(e: &mut Emulator) -> Option<StateFn> {
    e.updates_tx
        .send(AppUpdate::State(EmuState::Paused))
        .expect("could not update emulator state to paused");

    match try_get_key(POLL_TIMEOUT) {
        None => Some(StateFn(paused)),
        Some(key) => match key {
            KeyCode::Char('p') => return Some(StateFn(normal)),
            KeyCode::Char('o') => {
                e.updates_tx.send(AppUpdate::Exit).unwrap();
                return None;
            }
            KeyCode::Char('i') => return Some(StateFn(debug)),
            _ => Some(StateFn(paused)),
        },
    }
}

fn debug(e: &mut Emulator) -> Option<StateFn> {
    e.updates_tx
        .send(AppUpdate::State(EmuState::Debug))
        .expect("could not update emulator state to debug");

    match try_get_key(POLL_TIMEOUT) {
        None => Some(StateFn(debug)),
        Some(key) => match key {
            KeyCode::Char('p') => return Some(StateFn(paused)),
            KeyCode::Char('o') => {
                e.updates_tx.send(AppUpdate::Exit).unwrap();
                return None;
            }
            KeyCode::Char('i') => return Some(StateFn(normal)),
            KeyCode::Enter => {
                let i = e.fetch_next_and_decode().expect("could not decode next instruction");
                exec(i, e).expect("failed while executing instruction");
                Some(StateFn(debug))
            }
            _ => return Some(StateFn(debug)),
        },
    }
}

pub fn try_get_key(timeout: time::Duration) -> Option<KeyCode> {
    if event::poll(timeout).unwrap() {
        if let Event::Key(key) = event::read().unwrap() {
            return Some(key.code);
        }
    }
    None
}
