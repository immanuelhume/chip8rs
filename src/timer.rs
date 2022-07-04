use std::sync::{Arc, Mutex};
use std::thread;
use std::time;

pub struct Timer(Arc<Mutex<u8>>);

impl Timer {
    pub fn new(init: u8, period: time::Duration) -> Self {
        let val = Arc::new(Mutex::new(init));
        let val_c = Arc::clone(&val);
        let _ = thread::spawn(move || loop {
            {
                let mut val = val_c.lock().unwrap();
                match *val {
                    0 => (),
                    x => *val = x - 1,
                }
            }
            thread::sleep(period);
        });
        Self(val)
    }

    pub fn set(&self, val: u8) {
        *self.0.lock().unwrap() = val;
    }

    pub fn get(&self) -> u8 {
        *self.0.lock().unwrap()
    }
}

#[cfg(test)]
mod test_timer {
    use crate::timer::Timer;
    use std::{thread, time};

    #[test]
    fn set_and_get() {
        let timer = Timer::new(0, time::Duration::MAX);
        timer.set(8);
        assert_eq!(timer.get(), 8);
    }

    #[test]
    fn it_decrements() {
        let timer = Timer::new(8, time::Duration::from_nanos(1));
        thread::sleep(time::Duration::from_millis(1));
        assert_eq!(timer.get(), 0);
    }
}
