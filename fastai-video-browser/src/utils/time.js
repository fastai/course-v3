const getMinutes = (seconds) =>
  (seconds.toFixed(0) / 60).toString().split('.')[0];

const secondsToTimestamp = (totalSeconds) => {
  let minutes = getMinutes(totalSeconds);
  let remainder = (totalSeconds.toFixed(0) % 60).toString();
  if (minutes.length < 2) minutes = `0${minutes}`;
  if (remainder.length < 2) remainder = `0${remainder}`;
  return `${minutes}:${remainder}`;
};

const timestampToSeconds = (moment) => {
  const [minutes, seconds] = moment.split(':');
  return Number(minutes) * 60 + Number(seconds);
};

export {
  getMinutes,
  secondsToTimestamp,
  timestampToSeconds,
}
