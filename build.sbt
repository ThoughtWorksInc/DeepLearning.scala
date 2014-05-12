organization := "com.qifun"

name := "stateless-future"

version := "0.2.3-SNAPSHOT"

libraryDependencies <+= (scalaVersion) { sv =>
  "org.scala-lang" % "scala-reflect" % sv
}

libraryDependencies += "junit" % "junit-dep" % "4.10" % "test"

libraryDependencies += "com.novocode" % "junit-interface" % "0.10" % "test"

scalacOptions ++= Seq("-optimize", "-unchecked", "-Xlint", "-feature")

scalacOptions <++= (scalaVersion) map { sv =>
  if (sv.startsWith("2.10.")) {
    Seq("-deprecation") // Fully compatible with 2.10.x 
  } else {
    Seq() // May use deprecated API in 2.11.x
  }
}

incOptions := incOptions.value.withNameHashing(true)

crossScalaVersions := Seq("2.10.4", "2.11.0")

description := "The rubost asynchronous programming facility for Scala that offers a direct API for working with Futures."

homepage := Some(url("https://github.com/Atry/stateless-future"))

startYear := Some(2014)

licenses := Seq("Apache License, Version 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"))

publishTo <<= (isSnapshot) { isSnapshot: Boolean =>
  if (isSnapshot)
    Some("snapshots" at "https://oss.sonatype.org/content/repositories/snapshots")
  else
    Some("releases" at "https://oss.sonatype.org/service/local/staging/deploy/maven2")
}

scmInfo := Some(ScmInfo(
  url("https://github.com/Atry/stateless-future"),
  "scm:git:git://github.com/Atry/stateless-future.git",
  Some("scm:git:git@github.com:Atry/stateless-future.git")))

pomExtra :=
  <developers>
    <developer>
      <id>Atry</id>
      <name>杨博</name>
      <timezone>+8</timezone>
      <email>pop.atry@gmail.com</email>
    </developer>
  </developers>
